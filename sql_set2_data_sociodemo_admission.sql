WITH cohort AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM public.temp_notes_filtered
),
base AS (
  SELECT
    c.subject_id,
    c.hadm_id,

    p.gender,
    (p.anchor_age + (EXTRACT(YEAR FROM a.admittime)::int - p.anchor_year)) AS age_at_admit,

    a.race,
    a.admission_type,
    EXTRACT(YEAR FROM a.admittime)::int AS admit_year,
    a.admittime,
    a.dischtime,

    i.stay_id        AS icu_stay_id,
    i.first_careunit AS icu_type,
    i.intime         AS icu_intime

  FROM cohort c
  JOIN mimiciv_hosp.admissions a
    ON a.hadm_id = c.hadm_id
  JOIN mimiciv_hosp.patients p
    ON p.subject_id = c.subject_id
  LEFT JOIN mimiciv_icu.icustays i
    ON i.subject_id = c.subject_id
   AND i.hadm_id   = c.hadm_id
),
dedup AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id, hadm_id
      ORDER BY icu_intime NULLS LAST, icu_stay_id NULLS LAST
    ) AS rn
  FROM base
)
SELECT
  subject_id, hadm_id,
  gender, age_at_admit,
  race, admission_type, admit_year,
  admittime, dischtime,
  icu_stay_id, icu_type, icu_intime
FROM dedup
WHERE rn = 1;