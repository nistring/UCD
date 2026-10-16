WITH stays AS (
    SELECT i.stay_id, i.subject_id, i.hadm_id, i.intime, i.outtime
    FROM icu.icustays i
    QUALIFY row_number() OVER (PARTITION BY i.subject_id ORDER BY i.intime) = 1
),
ages AS (
    SELECT s.*, p.anchor_age + (year(a.admittime) - p.anchor_year) AS age, p.gender,
           a.hospital_expire_flag, a.deathtime
    FROM stays s
    JOIN hosp.admissions a ON a.hadm_id = s.hadm_id
    JOIN hosp.patients p ON p.subject_id = s.subject_id
)
SELECT
    stay_id AS unit_id,
    subject_id AS subject_key,
    intime AS index_time,
    outtime AS unit_end_time,
    age >= 18 AS incl_adult,
    CASE WHEN age >= 65 THEN 1 ELSE 0 END AS age_65,
    CASE WHEN gender = 'M' THEN 1 ELSE 0 END AS male,
    hospital_expire_flag AS outcome,
    CASE WHEN deathtime IS NOT NULL THEN date_diff('hour', intime, deathtime) END AS outcome_time_h
FROM ages
