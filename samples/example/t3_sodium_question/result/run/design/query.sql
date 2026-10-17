WITH icu_stays AS (
    SELECT
        icu.subject_id,
        icu.hadm_id,
        icu.stay_id,
        icu.intime,
        icu.outtime,
        pat.anchor_age,
        pat.anchor_year,
        pat.gender,
        adm.admittime,
        adm.admission_type,
        adm.hospital_expire_flag,
        adm.deathtime
    FROM icu.icustays icu
    JOIN hosp.admissions adm
      ON icu.hadm_id = adm.hadm_id
    JOIN hosp.patients pat
      ON icu.subject_id = pat.subject_id
    QUALIFY row_number() OVER (PARTITION BY icu.subject_id ORDER BY icu.intime) = 1
),
sodium_first AS (
    SELECT
        le.hadm_id,
        le.charttime,
        le.valuenum,
        row_number() OVER (
            PARTITION BY le.hadm_id
            ORDER BY le.charttime
        ) AS rn
    FROM hosp.labevents le
    JOIN icu.icustays icu
      ON le.hadm_id = icu.hadm_id
    WHERE le.itemid IN (50824, 50983)
      AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN icu.intime AND icu.intime + INTERVAL 24 HOUR
),
cohort AS (
    SELECT
        s.subject_id,
        s.hadm_id,
        s.stay_id,
        s.intime AS index_time,
        s.outtime AS unit_end_time,
        (s.anchor_age + (EXTRACT(YEAR FROM s.admittime) - s.anchor_year)) AS age,
        s.gender AS sex,
        CASE
            WHEN s.admission_type IN ('EW EMER.', 'DIRECT EMER.') THEN 'EMERGENCY'
            WHEN s.admission_type = 'URGENT' THEN 'URGENT'
            WHEN s.admission_type IN ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION') THEN 'ELECTIVE_SURGICAL'
            ELSE 'OTHER'
        END AS admission_type,
        s.hospital_expire_flag AS outcome,
        CASE
            WHEN s.anchor_age >= 18 THEN TRUE
            ELSE FALSE
        END AS incl_adult,
        CASE
            WHEN sf.charttime IS NOT NULL THEN TRUE
            ELSE FALSE
        END AS incl_has_sodium_24h,
        CASE
            WHEN sf.valuenum < 135 THEN TRUE
            ELSE FALSE
        END AS low_sodium,
        CASE
            WHEN s.deathtime IS NOT NULL
                 AND s.deathtime <= s.outtime THEN
                date_diff('hour', s.intime, s.deathtime)
            ELSE NULL
        END AS outcome_time_h,
        CASE
            WHEN s.deathtime IS NULL
                 OR s.deathtime > s.intime + INTERVAL 24 HOUR THEN TRUE
            ELSE FALSE
        END AS incl_alive_24h
    FROM icu_stays s
    LEFT JOIN sodium_first sf
      ON s.hadm_id = sf.hadm_id
     AND sf.rn = 1
     AND sf.charttime BETWEEN s.intime AND s.intime + INTERVAL 24 HOUR
)
SELECT
    subject_id AS subject_key,
    hadm_id,
    stay_id AS unit_id,
    index_time,
    unit_end_time,
    age,
    sex,
    admission_type,
    outcome,
    incl_adult,
    incl_has_sodium_24h,
    low_sodium,
    incl_alive_24h,
    outcome_time_h
FROM cohort
