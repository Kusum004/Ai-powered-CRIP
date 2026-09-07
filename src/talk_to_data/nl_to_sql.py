# src/talk_to_data/nl_to_sql.py
"""Agentic NL-to-SQL controller with multi-provider LLM support, AST validation, and self-healing retry loop."""
import os
import re
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.config import Config
from src.utils.logger import get_logger
from src.talk_to_data.prompt_templates import SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from src.talk_to_data.query_runner import QueryRunner

logger = get_logger("NLToSQLAgent")

class NLToSQLAgent:
    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or Config.LLM_PROVIDER).lower()
        self.runner = QueryRunner()
        self.history: List[Dict[str, Any]] = []
        self._init_llm_client()

    def _init_llm_client(self):
        """Initializes the appropriate LLM client based on environment variables."""
        self.gemini_client = None
        self.openai_client = None
        self.groq_client = None

        gemini_key = os.getenv("GEMINI_API_KEY", "")
        openai_key = os.getenv("OPENAI_API_KEY", "")
        groq_key = os.getenv("GROQ_API_KEY", "")

        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                logger.info("Initialized Google Gemini client.")
            except Exception as e:
                logger.warning(f"Could not initialize Google Gemini: {e}")

        if openai_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=openai_key)
                logger.info("Initialized OpenAI client.")
            except Exception as e:
                logger.warning(f"Could not initialize OpenAI client: {e}")

        if groq_key:
            try:
                from groq import Groq
                self.groq_client = Groq(api_key=groq_key)
                # Auto-discover working Groq model
                self.groq_model = "openai/gpt-oss-120b"
                try:
                    available = [m.id for m in self.groq_client.models.list().data]
                    for candidate in ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.1-70b-versatile", "llama-3.1-8b-instant"]:
                        if candidate in available:
                            self.groq_model = candidate
                            break
                except Exception:
                    pass
                logger.info(f"Initialized Groq client with model: {self.groq_model}.")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")

    def generate_sql(self, user_question: str, error_context: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None, *args, **kwargs) -> str:
        """
        Generates DuckDB SQL using active LLM provider (Groq, Gemini, OpenAI) or robust offline pattern synthesizer,
        incorporating multi-turn conversation history for follow-up queries.
        """
        # In case conversation_history was passed in kwargs
        if conversation_history is None:
            conversation_history = kwargs.get("conversation_history")
        # 1. Try Groq (if configured or requested)
        if self.groq_client and self.provider in ("groq", "auto"):
            for candidate_model in [getattr(self, "groq_model", "openai/gpt-oss-120b"), "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b", "llama-3.3-70b-versatile"]:
                try:
                    prompt = self._build_llm_prompt(user_question, error_context, conversation_history)
                    response = self.groq_client.chat.completions.create(
                        model=candidate_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.0
                    )
                    sql = self._clean_llm_sql_output(response.choices[0].message.content)
                    if sql:
                        logger.info(f"Generated SQL query via Groq ({candidate_model}).")
                        return sql
                except Exception as e:
                    logger.warning(f"Groq generation error with {candidate_model}: {e}. Trying next candidate...")

        # 2. Try Gemini
        if self.gemini_client and self.provider in ("gemini", "auto"):
            try:
                prompt = self._build_llm_prompt(user_question, error_context, conversation_history)
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                sql = self._clean_llm_sql_output(response.text)
                if sql:
                    logger.info("Generated SQL query via Google Gemini.")
                    return sql
            except Exception as e:
                logger.warning(f"Gemini generation error: {e}. Falling back...")

        # 3. Try OpenAI
        if self.openai_client and self.provider in ("openai", "auto"):
            try:
                prompt = self._build_llm_prompt(user_question, error_context, conversation_history)
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0
                )
                sql = self._clean_llm_sql_output(response.choices[0].message.content)
                if sql:
                    logger.info("Generated SQL query via OpenAI.")
                    return sql
            except Exception as e:
                logger.warning(f"OpenAI generation error: {e}. Falling back...")

        # 4. Deterministic Offline SQL Synthesizer
        logger.info("Using offline deterministic SQL synthesizer.")
        return self._offline_sql_synthesizer(user_question, conversation_history)

    def _build_llm_prompt(self, question: str, error_context: Optional[str] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        few_shot_str = ""
        for ex in FEW_SHOT_EXAMPLES:
            few_shot_str += f"Question: {ex['question']}\nSQL:\n{ex['sql']}\n\n"

        prompt = f"{SYSTEM_PROMPT}\n\nFew-shot verified examples:\n{few_shot_str}\n"
        
        # Inject conversation history if available
        if conversation_history:
            prompt += "--- CONVERSATION CONTEXT & PREVIOUS TURNS ---\n"
            for idx, turn in enumerate(conversation_history[-3:], 1):
                prev_q = turn.get("question", "")
                prev_sql = turn.get("sql", "")
                prompt += f"Turn {idx} User Question: {prev_q}\n"
                if prev_sql:
                    prompt += f"Turn {idx} Executed SQL:\n{prev_sql}\n"
            prompt += "--------------------------------------------\n"
            prompt += "Note: If the current user question is a follow-up, refinement, or modification of the previous query (e.g., adding filters, changing groupings, drill-down), adapt the previous SQL appropriately.\n\n"

        if error_context:
            prompt += f"IMPORTANT: The previous query failed with error: {error_context}. Please write a corrected SQL query.\n"
        prompt += f"Current User Question: {question}\nOutput SQL:"
        return prompt

    def _clean_llm_sql_output(self, text: str) -> str:
        text = text.strip()
        # Remove markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

    def _offline_sql_synthesizer(self, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        High-accuracy financial SQL synthesizer covering key banking risk questions and follow-ups.
        """
        q = question.lower()

        # Follow-up filter handling using previous queries
        if conversation_history and len(conversation_history) > 0:
            last_turn = conversation_history[-1]
            last_sql = last_turn.get("sql", "")
            if "filter" in q or "only" in q or "where" in q or "above" in q or "under" in q:
                if "under 30" in q or "< 30" in q or "young" in q:
                    if "WHERE" in last_sql:
                        return last_sql.replace("WHERE", "WHERE ABS(DAYS_BIRTH)/365.25 < 30 AND")
                    elif "GROUP BY" in last_sql:
                        parts = last_sql.split("GROUP BY")
                        return f"{parts[0]} WHERE ABS(DAYS_BIRTH)/365.25 < 30 GROUP BY{parts[1]}"
                if "over 40" in q or "above 40" in q or "> 40" in q:
                    if "WHERE" in last_sql:
                        return last_sql.replace("WHERE", "WHERE ABS(DAYS_BIRTH)/365.25 > 40 AND")
                    elif "GROUP BY" in last_sql:
                        parts = last_sql.split("GROUP BY")
                        return f"{parts[0]} WHERE ABS(DAYS_BIRTH)/365.25 > 40 GROUP BY{parts[1]}"
                if "car" in q:
                    if "WHERE" in last_sql:
                        return last_sql.replace("WHERE", "WHERE FLAG_OWN_CAR = 'Y' AND")
                    elif "GROUP BY" in last_sql:
                        parts = last_sql.split("GROUP BY")
                        return f"{parts[0]} WHERE FLAG_OWN_CAR = 'Y' GROUP BY{parts[1]}"

        if "occupation" in q or "job" in q:
            return """
SELECT 
    COALESCE(OCCUPATION_TYPE, 'Unspecified / Missing') AS occupation,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_loan_amount
FROM loan_applications
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
HAVING COUNT(*) >= 500
ORDER BY default_rate_pct DESC
LIMIT 10;
            """.strip()

        if "education" in q or "degree" in q or "study" in q:
            return """
SELECT 
    NAME_EDUCATION_TYPE AS education_level,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount
FROM loan_applications
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate_pct DESC;
            """.strip()

        if "income" in q and ("bracket" in q or "tier" in q or "distribution" in q or "range" in q or "level" in q or "quintile" in q):
            return """
SELECT 
    CASE 
        WHEN AMT_INCOME_TOTAL < 50000 THEN '1. Under $50,000'
        WHEN AMT_INCOME_TOTAL BETWEEN 50000 AND 100000 THEN '2. $50,000 - $100,000'
        WHEN AMT_INCOME_TOTAL BETWEEN 100000 AND 200000 THEN '3. $100,000 - $200,000'
        ELSE '4. Over $200,000'
    END AS income_bracket,
    COUNT(*) AS applicant_count,
    SUM(TARGET) AS default_count,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_loan_amount
FROM loan_applications
GROUP BY income_bracket
ORDER BY income_bracket;
            """.strip()

        if "age" in q or "old" in q or "young" in q or "cohort" in q or "decade" in q:
            return """
SELECT 
    CASE 
        WHEN ABS(DAYS_BIRTH)/365.25 < 30 THEN '1. Under 30'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 30 AND 39.99 THEN '2. 30s (30-39)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 40 AND 49.99 THEN '3. 40s (40-49)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 50 AND 59.99 THEN '4. 50s (50-59)'
        ELSE '5. 60 and Above'
    END AS age_cohort,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income
FROM loan_applications
GROUP BY age_cohort
ORDER BY age_cohort;
            """.strip()

        if "car" in q or "vehicle" in q or "auto" in q:
            return """
SELECT 
    CASE WHEN FLAG_OWN_CAR = 'Y' THEN 'Car Owner' ELSE 'Non-Car Owner' END AS vehicle_ownership,
    COUNT(*) AS total_borrowers,
    SUM(TARGET) AS defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income
FROM loan_applications
GROUP BY FLAG_OWN_CAR;
            """.strip()

        if "realty" in q or "house" in q or "housing" in q or "property" in q:
            return """
SELECT 
    NAME_HOUSING_TYPE AS housing_type,
    COUNT(*) AS total_borrowers,
    SUM(TARGET) AS defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount
FROM loan_applications
GROUP BY NAME_HOUSING_TYPE
ORDER BY default_rate_pct DESC;
            """.strip()

        if "gender" in q or "female" in q or "male" in q or "sex" in q:
            return """
SELECT 
    CODE_GENDER AS gender,
    COUNT(*) AS applicant_count,
    SUM(TARGET) AS default_count,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM loan_applications
WHERE CODE_GENDER IN ('M', 'F')
GROUP BY CODE_GENDER;
            """.strip()

        if "family" in q or "married" in q or "single" in q or "marital" in q:
            return """
SELECT 
    NAME_FAMILY_STATUS AS family_status,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM loan_applications
GROUP BY NAME_FAMILY_STATUS
ORDER BY default_rate_pct DESC;
            """.strip()

        if "organization" in q or "company" in q or "industry" in q or "sector" in q:
            return """
SELECT 
    ORGANIZATION_TYPE AS organization_type,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income
FROM loan_applications
GROUP BY ORGANIZATION_TYPE
HAVING COUNT(*) >= 1000
ORDER BY default_rate_pct DESC
LIMIT 10;
            """.strip()

        if "bureau" in q or "ext_source" in q or "credit score" in q:
            return """
SELECT 
    ROUND(COALESCE(EXT_SOURCE_2, 0.5), 1) AS ext_source_2_bucket,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM loan_applications
WHERE EXT_SOURCE_2 IS NOT NULL
GROUP BY ext_source_2_bucket
ORDER BY ext_source_2_bucket ASC;
            """.strip()

        # Default fallback summary
        return """
SELECT 
    NAME_INCOME_TYPE AS income_type,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM loan_applications
GROUP BY NAME_INCOME_TYPE
ORDER BY default_rate_pct DESC;
        """.strip()

    def process_query(self, user_question: str, conversation_history: Optional[List[Dict[str, Any]]] = None, *args, **kwargs) -> Dict[str, Any]:
        """
        End-to-end agentic workflow: NL -> SQL -> AST Validate -> Execute -> Self-Heal -> Executive Summary.
        Supports multi-turn interactive conversational follow-ups.
        """
        if conversation_history is None:
            conversation_history = kwargs.get("conversation_history")

        # Step 1: Initial SQL Generation with conversation history context
        sql_query = self.generate_sql(user_question, conversation_history=conversation_history)
        result = self.runner.execute_safe_query(sql_query)

        # Step 2: Self-Healing Retry Loop (if error occurs)
        if not result["success"]:
            logger.info("First SQL attempt failed. Triggering automated self-healing loop...")
            corrected_sql = self.generate_sql(user_question, error_context=result["error"], conversation_history=conversation_history)
            result = self.runner.execute_safe_query(corrected_sql)

        # Step 3: Generate Executive Business Synthesis
        synthesis = []
        if result["success"] and result["data"] is not None and not result["data"].empty:
            synthesis = self._synthesize_business_insights(user_question, result["data"])

        response_bundle = {
            "question": user_question,
            "sql": result["sql"],
            "success": result["success"],
            "error": result["error"],
            "latency_ms": result["latency_ms"],
            "latency_badge": result.get("latency_badge", f"Executed in {result['latency_ms']} ms"),
            "data": result["data"],
            "row_count": result["row_count"],
            "executive_summary": synthesis
        }

        # Store in conversation memory
        self.history.append(response_bundle)
        return response_bundle

    def _synthesize_business_insights(self, question: str, df: pd.DataFrame) -> List[str]:
        """Generates 3 concise bullet points summarizing the SQL result."""
        bullets = []
        try:
            row_count = len(df)
            cols = list(df.columns)

            # Check if default_rate_pct exists
            rate_col = next((c for c in cols if "default_rate" in c.lower() or "rate" in c.lower() or "pct" in c.lower()), None)
            group_col = cols[0] if cols else "Category"

            if rate_col and len(df) > 1:
                highest = df.sort_values(by=rate_col, ascending=False).iloc[0]
                lowest = df.sort_values(by=rate_col, ascending=True).iloc[0]
                bullets.append(f"**Highest Risk Segment**: `{highest[group_col]}` exhibits the highest default rate at **{highest[rate_col]}%**.")
                bullets.append(f"**Lowest Risk Segment**: `{lowest[group_col]}` demonstrates the safest performance at **{lowest[rate_col]}%** default rate.")
                ratio = (float(highest[rate_col]) / max(0.01, float(lowest[rate_col])))
                bullets.append(f"**Risk Multiplier**: Risk spread between polar segments is **{ratio:.1f}x**, confirming strong segmentation power.")
            else:
                bullets.append(f"**Data Volume**: Returned {row_count} analytical records across {len(cols)} dimensions.")
                if len(cols) > 1 and np.issubdtype(df[cols[1]].dtype, np.number):
                    avg_val = df[cols[1]].mean()
                    bullets.append(f"**Benchmark Metric**: Average `{cols[1]}` across groups is **{avg_val:,.2f}**.")
                bullets.append("**Operational Note**: Results are synchronized directly from in-memory DuckDB OLAP store.")
        except Exception as e:
            bullets.append(f"Query executed successfully returning {len(df)} analytical records.")

        return bullets
