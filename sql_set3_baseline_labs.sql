-- ============================================
-- SET 3 (labs): baseline labs near ICU admit
-- Cohort: public.temp_notes_filtered
-- Time zero: first ICU intime per (subject_id, hadm_id)
-- Window: +/- 6 hours around ICU intime
-- Output: public.set_3_labs_baseline_icu
-- Fixes:
--   - Lactate: hardcode itemid=50813 (mmol/L) to avoid LDH contamination
--   - FiO2: use chartevents itemid=223835 (Inspired O2 Fraction), not labevents
-- ============================================

DROP TABLE IF EXISTS public.set_3_labs_baseline_icu;

CREATE TABLE public.set_3_labs_baseline_icu AS
WITH cohort AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM public.temp_notes_filtered
),
idx AS (
  -- Primera ICU por (subject_id, hadm_id)
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

lab_itemids AS (
  -- Lactate (VALIDATED): itemid=50813, mmol/L
  SELECT 'lactate' AS lab, 50813::int AS itemid

  UNION ALL
  SELECT DISTINCT 'creatinine' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE d.label ILIKE 'creatinine%'
    AND d.fluid ILIKE '%blood%'

  UNION ALL
  SELECT DISTINCT 'bun' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE (d.label ILIKE '%urea nitrogen%' OR d.label ILIKE 'bun%')
    AND d.fluid ILIKE '%blood%'

  UNION ALL
  SELECT DISTINCT 'bilirubin_total' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE d.label ILIKE '%bilirubin%'
    AND d.label ILIKE '%total%'
    AND d.fluid ILIKE '%blood%'

  UNION ALL
  SELECT DISTINCT 'platelets' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE (d.label ILIKE '%platelet%' AND d.label ILIKE '%count%')
    AND d.fluid ILIKE '%blood%'

  UNION ALL
  SELECT DISTINCT 'wbc' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE (d.label ILIKE 'wbc%' OR d.label ILIKE '%white blood%')
    AND d.fluid ILIKE '%blood%'

  UNION ALL
  -- Blood gas: PaO2 (pO2/PO2)
  SELECT DISTINCT 'pao2' AS lab, d.itemid
  FROM mimiciv_hosp.d_labitems d
  WHERE (d.label ILIKE '%pO2%' OR d.label ILIKE '%po2%')
    AND d.fluid ILIKE '%blood%'
    AND d.category ILIKE '%blood gas%'
),

lab_events AS (
  -- Labs dentro de la ventana y rank por cercanía a ICU intime
  SELECT
    idx.subject_id,
    idx.hadm_id,
    idx.index_stay_id,
    idx.index_icu_intime,
    li.lab,
    le.charttime,
    le.valuenum,
    le.valueuom,
    ROW_NUMBER() OVER (
      PARTITION BY idx.subject_id, idx.hadm_id, li.lab
      ORDER BY ABS(EXTRACT(EPOCH FROM (le.charttime - idx.index_icu_intime))) ASC
    ) AS rn
  FROM idx
  JOIN lab_itemids li ON TRUE
  JOIN mimiciv_hosp.labevents le
    ON le.hadm_id = idx.hadm_id
   AND le.itemid  = li.itemid
  WHERE le.valuenum IS NOT NULL
    AND le.charttime BETWEEN idx.index_icu_intime - INTERVAL '6 hours'
                        AND idx.index_icu_intime + INTERVAL '6 hours'
    -- lactate strict unit (optional but recommended)
    AND (
      li.lab <> 'lactate'
      OR le.valueuom IS NULL
      OR le.valueuom = 'mmol/L'
    )
),

pivot AS (
  SELECT
    idx.subject_id,
    idx.hadm_id,
    idx.index_stay_id,
    idx.index_icu_intime,
    idx.index_icu_type,

    MAX(le.valuenum)  FILTER (WHERE le.lab='lactate'         AND le.rn=1) AS lactate,
    MAX(le.valueuom)  FILTER (WHERE le.lab='lactate'         AND le.rn=1) AS lactate_uom,
    MAX(le.charttime) FILTER (WHERE le.lab='lactate'         AND le.rn=1) AS lactate_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='creatinine'      AND le.rn=1) AS creatinine,
    MAX(le.charttime) FILTER (WHERE le.lab='creatinine'      AND le.rn=1) AS creatinine_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='bun'             AND le.rn=1) AS bun,
    MAX(le.charttime) FILTER (WHERE le.lab='bun'             AND le.rn=1) AS bun_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='bilirubin_total' AND le.rn=1) AS bilirubin_total,
    MAX(le.charttime) FILTER (WHERE le.lab='bilirubin_total' AND le.rn=1) AS bilirubin_total_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='platelets'       AND le.rn=1) AS platelets,
    MAX(le.charttime) FILTER (WHERE le.lab='platelets'       AND le.rn=1) AS platelets_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='wbc'             AND le.rn=1) AS wbc,
    MAX(le.charttime) FILTER (WHERE le.lab='wbc'             AND le.rn=1) AS wbc_time,

    MAX(le.valuenum)  FILTER (WHERE le.lab='pao2'            AND le.rn=1) AS pao2,
    MAX(le.charttime) FILTER (WHERE le.lab='pao2'            AND le.rn=1) AS pao2_time

  FROM idx
  LEFT JOIN lab_events le
    ON le.subject_id = idx.subject_id
   AND le.hadm_id    = idx.hadm_id
  GROUP BY
    idx.subject_id, idx.hadm_id, idx.index_stay_id, idx.index_icu_intime, idx.index_icu_type
),

-- FiO2 from ICU chartevents (VALIDATED approach)
fio2_near AS (
  SELECT
    idx.subject_id,
    idx.hadm_id,
    ce.charttime,
    ce.valuenum,
    ROW_NUMBER() OVER (
      PARTITION BY idx.subject_id, idx.hadm_id
      ORDER BY ABS(EXTRACT(EPOCH FROM (ce.charttime - idx.index_icu_intime))) ASC
    ) AS rn
  FROM idx
  JOIN mimiciv_icu.chartevents ce
    ON ce.stay_id = idx.index_stay_id
   AND ce.itemid  = 223835  -- Inspired O2 Fraction
  WHERE ce.valuenum IS NOT NULL
    AND ce.charttime BETWEEN idx.index_icu_intime - INTERVAL '6 hours'
                        AND idx.index_icu_intime + INTERVAL '6 hours'
),

final AS (
  SELECT
    p.*,

    f.valuenum  AS fio2_raw,
    f.charttime AS fio2_time,

    -- Normalize FiO2: if 21-100 => /100, if already 0-1 keep
    CASE
      WHEN f.valuenum IS NULL THEN NULL
      WHEN f.valuenum > 1 THEN f.valuenum / 100.0
      ELSE f.valuenum
    END AS fio2_frac

  FROM pivot p
  LEFT JOIN fio2_near f
    ON f.subject_id = p.subject_id
   AND f.hadm_id    = p.hadm_id
   AND f.rn = 1
)

SELECT
  final.*,

  CASE
    WHEN final.pao2 IS NULL OR final.fio2_frac IS NULL THEN NULL
    WHEN final.fio2_frac <= 0 THEN NULL
    ELSE final.pao2 / final.fio2_frac
  END AS pf_ratio

FROM final;