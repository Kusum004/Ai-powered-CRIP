# src/talk_to_data/prompt_templates.py
"""SQL schema injection, column definitions, and few-shot analytical prompt templates."""

TABLE_SCHEMA = """
CREATE TABLE loan_applications (
    SK_ID_CURR INTEGER,
    TARGET INTEGER,                     -- 0: Loan repaid, 1: Default / payment difficulty
    NAME_CONTRACT_TYPE VARCHAR,         -- Cash loans, Revolving loans
    CODE_GENDER VARCHAR,                -- M, F, XNA
    FLAG_OWN_CAR VARCHAR,               -- Y, N
    FLAG_OWN_REALTY VARCHAR,            -- Y, N
    CNT_CHILDREN INTEGER,               -- Number of children
    AMT_INCOME_TOTAL DOUBLE,            -- Total annual income ($)
    AMT_CREDIT DOUBLE,                  -- Credit amount of the loan ($)
    AMT_ANNUITY DOUBLE,                 -- Monthly loan payment / annuity ($)
    AMT_GOODS_PRICE DOUBLE,             -- Price of the goods for which the loan is given ($)
    NAME_INCOME_TYPE VARCHAR,           -- Working, Commercial associate, Pensioner, State servant
    NAME_EDUCATION_TYPE VARCHAR,        -- Higher education, Secondary / secondary special, Incomplete higher, Lower secondary, Academic degree
    NAME_FAMILY_STATUS VARCHAR,         -- Married, Single / not married, Civil marriage, Separated, Widow
    NAME_HOUSING_TYPE VARCHAR,          -- House / apartment, With parents, Municipal apartment, Rented apartment, Office apartment, Co-op apartment
    REGION_POPULATION_RELATIVE DOUBLE,  -- Normalized population of region
    DAYS_BIRTH INTEGER,                 -- Client age in days at the time of application (negative integer, e.g. -12000)
    DAYS_EMPLOYED INTEGER,              -- How many days before the application the person started current employment (negative integer, 365243 is NaN)
    DAYS_REGISTRATION DOUBLE,           -- How many days before application client changed registration
    DAYS_ID_PUBLISH INTEGER,            -- How many days before application client changed identification document
    OWN_CAR_AGE DOUBLE,                 -- Age of client's car in years
    OCCUPATION_TYPE VARCHAR,            -- Laborers, Core staff, Sales staff, Managers, Drivers, etc.
    CNT_FAM_MEMBERS DOUBLE,             -- Number of family members
    REGION_RATING_CLIENT INTEGER,       -- Rating of the region (1, 2, 3)
    REGION_RATING_CLIENT_W_CITY INTEGER,-- Rating of the region with city (1, 2, 3)
    ORGANIZATION_TYPE VARCHAR,          -- Type of organization where client works (Business Entity Type 3, Self-employed, etc.)
    EXT_SOURCE_1 DOUBLE,                -- Normalized score from external data source 1 (0 to 1)
    EXT_SOURCE_2 DOUBLE,                -- Normalized score from external data source 2 (0 to 1)
    EXT_SOURCE_3 DOUBLE,                -- Normalized score from external data source 3 (0 to 1)
    DEF_30_CNT_SOCIAL_CIRCLE DOUBLE,    -- Number of client's social surroundings defaulted on 30 DPD
    DEF_60_CNT_SOCIAL_CIRCLE DOUBLE,    -- Number of client's social surroundings defaulted on 60 DPD
    DAYS_LAST_PHONE_CHANGE DOUBLE,      -- How many days before application client changed phone
    AMT_REQ_CREDIT_BUREAU_YEAR DOUBLE   -- Number of inquiries to Credit Bureau about client one year before application
);
"""

FEW_SHOT_EXAMPLES = [
    {
        "question": "What is the default rate across different annual income brackets (<$50k, $50k-$100k, >$100k)?",
        "sql": """
SELECT 
    CASE 
        WHEN AMT_INCOME_TOTAL < 50000 THEN 'Under $50,000'
        WHEN AMT_INCOME_TOTAL BETWEEN 50000 AND 100000 THEN '$50,000 - $100,000'
        ELSE 'Over $100,000'
    END AS income_bracket,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_loan_amount
FROM loan_applications
GROUP BY income_bracket
ORDER BY default_rate_pct DESC;
        """.strip()
    },
    {
        "question": "Show top 5 highest risk occupations with at least 500 applicants.",
        "sql": """
SELECT 
    COALESCE(OCCUPATION_TYPE, 'Unspecified') AS occupation,
    COUNT(*) AS applicant_count,
    SUM(TARGET) AS default_count,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income
FROM loan_applications
WHERE OCCUPATION_TYPE IS NOT NULL
GROUP BY OCCUPATION_TYPE
HAVING COUNT(*) >= 500
ORDER BY default_rate_pct DESC
LIMIT 5;
        """.strip()
    },
    {
        "question": "Calculate average loan annuity and credit amount grouped by education level.",
        "sql": """
SELECT 
    NAME_EDUCATION_TYPE AS education_level,
    COUNT(*) AS total_applicants,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount,
    ROUND(AVG(AMT_ANNUITY), 2) AS avg_monthly_annuity,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct
FROM loan_applications
GROUP BY NAME_EDUCATION_TYPE
ORDER BY avg_credit_amount DESC;
        """.strip()
    },
    {
        "question": "Compare default rate and loan amount between car owners and non-car owners.",
        "sql": """
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
    },
    {
        "question": "What is the risk profile across age cohorts (under 30, 30s, 40s, 50s, 60+)?",
        "sql": """
SELECT 
    CASE 
        WHEN ABS(DAYS_BIRTH)/365.25 < 30 THEN 'Under 30'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 30 AND 39.99 THEN '30s (30-39)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 40 AND 49.99 THEN '40s (40-49)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 50 AND 59.99 THEN '50s (50-59)'
        ELSE '60 and Above'
    END AS age_bracket,
    COUNT(*) AS applicant_count,
    SUM(TARGET) AS default_count,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income
FROM loan_applications
GROUP BY age_bracket
ORDER BY default_rate_pct DESC;
        """.strip()
    }
]

SYSTEM_PROMPT = f"""You are an expert Chief Risk Officer and Senior Quantitative SQL Architect specializing in the Home Credit loan dataset.
Your job is to translate financial, underwriting, and risk intelligence questions into safe, optimized, single DuckDB SQL queries.

Database Schema:
{TABLE_SCHEMA}

Key SQL Domain Guidelines:
1. The table name is `loan_applications`.
2. `TARGET` is 1 for default/delinquency and 0 for repaid. Use `ROUND(AVG(TARGET) * 100.0, 2)` to calculate default rate percentage.
3. `DAYS_BIRTH` is negative days since birth. Convert to age years with `ABS(DAYS_BIRTH)/365.25`.
4. `DAYS_EMPLOYED` is negative days. Ignore 365243 as it represents unemployed/pensioner.
5. All queries MUST be pure `SELECT` statements. Never output `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or `CREATE`.
6. Return ONLY the raw SQL query. Do not wrap in markdown quotes or add conversational filler.
"""
