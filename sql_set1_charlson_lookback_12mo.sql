-- ============================================
-- Charlson lookback 12m prior to ICU intime
-- ROBUST version: dedup source tables + dedup final
-- Output: public.set_2_charlson_prior12m_icu
-- Requires: mimiciv_derived.charlson (mimic-code)
-- ============================================

DROP TABLE IF EXISTS public.set_2_charlson_prior12m_icu;

CREATE TABLE public.set_2_charlson_prior12m_icu AS
WITH cohort AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM public.temp_notes_filtered
),
patients_u AS (
  SELECT DISTINCT ON (subject_id)
    subject_id, gender, anchor_age, anchor_year
  FROM mimiciv_hosp.patients
  ORDER BY subject_id, anchor_year DESC
),
admissions_u AS (
  SELECT DISTINCT ON (hadm_id)
    hadm_id, subject_id, admittime, dischtime, race, admission_type
  FROM mimiciv_hosp.admissions
  ORDER BY hadm_id, admittime
),
index_icu AS (
  SELECT DISTINCT ON (i.subject_id, i.hadm_id)
    i.subject_id,
    i.hadm_id,
    i.stay_id        AS index_stay_id,
    i.intime         AS index_icu_intime,
    i.first_careunit AS index_icu_type
  FROM mimiciv_icu.icustays i
  JOIN cohort c
    ON c.subject_id = i.subject_id
   AND c.hadm_id   = i.hadm_id
  ORDER BY i.subject_id, i.hadm_id, i.intime
),
index_age AS (
  SELECT
    idx.subject_id,
    idx.hadm_id,
    idx.index_stay_id,
    idx.index_icu_intime,
    idx.index_icu_type,
    (p.anchor_age + (EXTRACT(YEAR FROM idx.index_icu_intime)::int - p.anchor_year)) AS age_at_icu_intime,
    CASE
      WHEN (p.anchor_age + (EXTRACT(YEAR FROM idx.index_icu_intime)::int - p.anchor_year)) <= 50 THEN 0
      WHEN (p.anchor_age + (EXTRACT(YEAR FROM idx.index_icu_intime)::int - p.anchor_year)) <= 60 THEN 1
      WHEN (p.anchor_age + (EXTRACT(YEAR FROM idx.index_icu_intime)::int - p.anchor_year)) <= 70 THEN 2
      WHEN (p.anchor_age + (EXTRACT(YEAR FROM idx.index_icu_intime)::int - p.anchor_year)) <= 80 THEN 3
      ELSE 4
    END AS age_score
  FROM index_icu idx
  JOIN patients_u p
    ON p.subject_id = idx.subject_id
),
prior_adm AS (
  -- hadm previos (excluye index hadm) con dischtime en los 12m previos a ICU intime
  SELECT
    ia.subject_id,
    ia.hadm_id AS index_hadm_id,
    a.hadm_id  AS prior_hadm_id
  FROM index_age ia
  JOIN admissions_u a
    ON a.subject_id = ia.subject_id
  WHERE a.hadm_id <> ia.hadm_id
    AND a.dischtime < ia.index_icu_intime
    AND a.dischtime >= (ia.index_icu_intime - INTERVAL '12 months')
),
agg_flags AS (
  SELECT
    pa.subject_id,
    pa.index_hadm_id AS hadm_id,
    COUNT(DISTINCT pa.prior_hadm_id) AS n_prior_hadm_12m,

    MAX(ch.myocardial_infarct)::int           AS myocardial_infarct,
    MAX(ch.congestive_heart_failure)::int     AS congestive_heart_failure,
    MAX(ch.peripheral_vascular_disease)::int  AS peripheral_vascular_disease,
    MAX(ch.cerebrovascular_disease)::int      AS cerebrovascular_disease,
    MAX(ch.dementia)::int                     AS dementia,
    MAX(ch.chronic_pulmonary_disease)::int    AS chronic_pulmonary_disease,
    MAX(ch.rheumatic_disease)::int            AS rheumatic_disease,
    MAX(ch.peptic_ulcer_disease)::int         AS peptic_ulcer_disease,
    MAX(ch.mild_liver_disease)::int           AS mild_liver_disease,
    MAX(ch.diabetes_without_cc)::int          AS diabetes_without_cc,
    MAX(ch.diabetes_with_cc)::int             AS diabetes_with_cc,
    MAX(ch.paraplegia)::int                   AS paraplegia,
    MAX(ch.renal_disease)::int                AS renal_disease,
    MAX(ch.malignant_cancer)::int             AS malignant_cancer,
    MAX(ch.severe_liver_disease)::int         AS severe_liver_disease,
    MAX(ch.metastatic_solid_tumor)::int       AS metastatic_solid_tumor,
    MAX(ch.aids)::int                         AS aids

  FROM prior_adm pa
  LEFT JOIN mimiciv_derived.charlson ch
    ON ch.hadm_id = pa.prior_hadm_id
  GROUP BY pa.subject_id, pa.index_hadm_id
),
final_raw AS (
  SELECT
    ia.subject_id,
    ia.hadm_id,
    ia.index_stay_id,
    ia.index_icu_intime,
    ia.index_icu_type,
    ia.age_at_icu_intime,
    ia.age_score,
    COALESCE(af.n_prior_hadm_12m, 0) AS n_prior_hadm_12m,

    COALESCE(af.myocardial_infarct, 0)          AS myocardial_infarct,
    COALESCE(af.congestive_heart_failure, 0)    AS congestive_heart_failure,
    COALESCE(af.peripheral_vascular_disease, 0) AS peripheral_vascular_disease,
    COALESCE(af.cerebrovascular_disease, 0)     AS cerebrovascular_disease,
    COALESCE(af.dementia, 0)                    AS dementia,
    COALESCE(af.chronic_pulmonary_disease, 0)   AS chronic_pulmonary_disease,
    COALESCE(af.rheumatic_disease, 0)           AS rheumatic_disease,
    COALESCE(af.peptic_ulcer_disease, 0)        AS peptic_ulcer_disease,
    COALESCE(af.mild_liver_disease, 0)          AS mild_liver_disease,
    COALESCE(af.diabetes_without_cc, 0)         AS diabetes_without_cc,
    COALESCE(af.diabetes_with_cc, 0)            AS diabetes_with_cc,
    COALESCE(af.paraplegia, 0)                  AS paraplegia,
    COALESCE(af.renal_disease, 0)               AS renal_disease,
    COALESCE(af.malignant_cancer, 0)            AS malignant_cancer,
    COALESCE(af.severe_liver_disease, 0)        AS severe_liver_disease,
    COALESCE(af.metastatic_solid_tumor, 0)      AS metastatic_solid_tumor,
    COALESCE(af.aids, 0)                        AS aids,

    (
      COALESCE(af.myocardial_infarct,0)
      + COALESCE(af.congestive_heart_failure,0)
      + COALESCE(af.peripheral_vascular_disease,0)
      + COALESCE(af.cerebrovascular_disease,0)
      + COALESCE(af.dementia,0)
      + COALESCE(af.chronic_pulmonary_disease,0)
      + COALESCE(af.rheumatic_disease,0)
      + COALESCE(af.peptic_ulcer_disease,0)
      + GREATEST(COALESCE(af.mild_liver_disease,0), 3 * COALESCE(af.severe_liver_disease,0))
      + GREATEST(2 * COALESCE(af.diabetes_with_cc,0), COALESCE(af.diabetes_without_cc,0))
      + GREATEST(2 * COALESCE(af.malignant_cancer,0), 6 * COALESCE(af.metastatic_solid_tumor,0))
      + 2 * COALESCE(af.paraplegia,0)
      + 2 * COALESCE(af.renal_disease,0)
      + 6 * COALESCE(af.aids,0)
    ) AS charlson_12m_prior_no_age,

    (
      ia.age_score
      + COALESCE(af.myocardial_infarct,0)
      + COALESCE(af.congestive_heart_failure,0)
      + COALESCE(af.peripheral_vascular_disease,0)
      + COALESCE(af.cerebrovascular_disease,0)
      + COALESCE(af.dementia,0)
      + COALESCE(af.chronic_pulmonary_disease,0)
      + COALESCE(af.rheumatic_disease,0)
      + COALESCE(af.peptic_ulcer_disease,0)
      + GREATEST(COALESCE(af.mild_liver_disease,0), 3 * COALESCE(af.severe_liver_disease,0))
      + GREATEST(2 * COALESCE(af.diabetes_with_cc,0), COALESCE(af.diabetes_without_cc,0))
      + GREATEST(2 * COALESCE(af.malignant_cancer,0), 6 * COALESCE(af.metastatic_solid_tumor,0))
      + 2 * COALESCE(af.paraplegia,0)
      + 2 * COALESCE(af.renal_disease,0)
      + 6 * COALESCE(af.aids,0)
    ) AS charlson_12m_prior_with_age
  FROM index_age ia
  LEFT JOIN agg_flags af
    ON af.subject_id = ia.subject_id
   AND af.hadm_id    = ia.hadm_id
),
dedup_final AS (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY subject_id, hadm_id
      ORDER BY index_icu_intime NULLS LAST, index_stay_id NULLS LAST
    ) AS rn
  FROM final_raw
)
SELECT *
FROM dedup_final
WHERE rn = 1;