-- sql/schema.sql
-- DuckDB Schema & Analytical Views for Credit Risk Intelligence Platform

CREATE TABLE IF NOT EXISTS loan_applications (
    SK_ID_CURR INTEGER PRIMARY KEY,
    TARGET INTEGER,
    NAME_CONTRACT_TYPE VARCHAR,
    CODE_GENDER VARCHAR,
    FLAG_OWN_CAR VARCHAR,
    FLAG_OWN_REALTY VARCHAR,
    CNT_CHILDREN INTEGER,
    AMT_INCOME_TOTAL DOUBLE,
    AMT_CREDIT DOUBLE,
    AMT_ANNUITY DOUBLE,
    AMT_GOODS_PRICE DOUBLE,
    NAME_INCOME_TYPE VARCHAR,
    NAME_EDUCATION_TYPE VARCHAR,
    NAME_FAMILY_STATUS VARCHAR,
    NAME_HOUSING_TYPE VARCHAR,
    REGION_POPULATION_RELATIVE DOUBLE,
    DAYS_BIRTH INTEGER,
    DAYS_EMPLOYED INTEGER,
    DAYS_REGISTRATION DOUBLE,
    DAYS_ID_PUBLISH INTEGER,
    OWN_CAR_AGE DOUBLE,
    OCCUPATION_TYPE VARCHAR,
    CNT_FAM_MEMBERS DOUBLE,
    REGION_RATING_CLIENT INTEGER,
    REGION_RATING_CLIENT_W_CITY INTEGER,
    ORGANIZATION_TYPE VARCHAR,
    EXT_SOURCE_1 DOUBLE,
    EXT_SOURCE_2 DOUBLE,
    EXT_SOURCE_3 DOUBLE,
    DEF_30_CNT_SOCIAL_CIRCLE DOUBLE,
    DEF_60_CNT_SOCIAL_CIRCLE DOUBLE,
    DAYS_LAST_PHONE_CHANGE DOUBLE,
    AMT_REQ_CREDIT_BUREAU_YEAR DOUBLE
);

-- Analytical View: Portfolio KPI Summary by Education Level
CREATE OR REPLACE VIEW v_education_risk_summary AS
SELECT 
    NAME_EDUCATION_TYPE AS education_level,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_annual_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_loan_amount,
    ROUND(AVG(AMT_ANNUITY), 2) AS avg_annuity
FROM loan_applications
GROUP BY NAME_EDUCATION_TYPE
ORDER BY default_rate_pct DESC;

-- Analytical View: High-Risk Occupation Ranking
CREATE OR REPLACE VIEW v_occupation_risk_ranking AS
SELECT 
    COALESCE(OCCUPATION_TYPE, 'Unspecified / Missing') AS occupation,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_INCOME_TOTAL), 2) AS avg_income,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit
FROM loan_applications
GROUP BY OCCUPATION_TYPE
HAVING COUNT(*) >= 500
ORDER BY default_rate_pct DESC;

-- Analytical View: Risk by Age Cohorts
CREATE OR REPLACE VIEW v_age_cohort_risk AS
SELECT 
    CASE 
        WHEN ABS(DAYS_BIRTH)/365.25 < 30 THEN 'Under 30'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 30 AND 39.99 THEN '30s (30-39)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 40 AND 49.99 THEN '40s (40-49)'
        WHEN ABS(DAYS_BIRTH)/365.25 BETWEEN 50 AND 59.99 THEN '50s (50-59)'
        ELSE '60 and Above'
    END AS age_cohort,
    COUNT(*) AS total_applicants,
    SUM(TARGET) AS total_defaulters,
    ROUND(AVG(TARGET) * 100.0, 2) AS default_rate_pct,
    ROUND(AVG(AMT_CREDIT), 2) AS avg_credit_amount
FROM loan_applications
GROUP BY age_cohort
ORDER BY default_rate_pct DESC;
