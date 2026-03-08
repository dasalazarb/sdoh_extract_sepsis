-- ============================================
-- SET 4: Early interventions around ICU admit
-- Cohort: public.temp_notes_filtered
-- Time zero: first ICU intime per (subject_id, hadm_id)
-- Window: +/- 6 hours (editable)
-- Output: public.set_4_early_interventions_icu
-- ============================================

DROP TABLE IF EXISTS public.set_4_early_interventions_icu;

CREATE TABLE public.set_4_early_interventions_icu AS
WITH cohort AS (
  SELECT DISTINCT subject_id, hadm_id
  FROM public.temp_notes_filtered
),
idx AS (
  -- primera ICU por (subject_id, hadm_id)
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
params AS (
  SELECT
    interval '6 hours' AS win
),

-- ----------------------------
-- Ventilation itemids from d_items
-- ----------------------------
o2_device_itemids AS (
  SELECT itemid
  FROM mimiciv_icu.d_items
  WHERE label ILIKE '%o2 delivery device%'
     OR label ILIKE '%oxygen delivery device%'
),
vent_mode_itemids AS (
  SELECT itemid
  FROM mimiciv_icu.d_items
  WHERE label ILIKE '%ventilator mode%'
     OR label ILIKE '%vent mode%'
),

-- ----------------------------
-- Main flags
-- ----------------------------
final AS (
  SELECT
    idx.subject_id,
    idx.hadm_id,
    idx.index_stay_id,
    idx.index_icu_intime,
    idx.index_icu_type,

    /* ----------------------------
       1) Vasopressors within window
       ---------------------------- */
    CASE WHEN EXISTS (
      SELECT 1
      FROM mimiciv_icu.inputevents ie
      JOIN mimiciv_icu.d_items di
        ON di.itemid = ie.itemid
      CROSS JOIN params p
      WHERE ie.stay_id = idx.index_stay_id
        AND ie.starttime >= idx.index_icu_intime - p.win
        AND ie.starttime <= idx.index_icu_intime + p.win
        AND (
          di.label ILIKE '%norepinephrine%' OR
          di.label ILIKE '%epinephrine%' OR
          di.label ILIKE '%vasopressin%' OR
          di.label ILIKE '%dopamine%' OR
          di.label ILIKE '%phenylephrine%'
        )
        AND COALESCE(ie.amount, 0) > 0
    ) THEN 1 ELSE 0 END AS vasopressor_within_6h,

    /* ----------------------------
       2) Invasive mechanical ventilation within window
          Based on:
          - O2 delivery device value = Endotracheal tube / Tracheostomy tube
          - Ventilator mode value in common invasive modes
       ---------------------------- */
    CASE WHEN EXISTS (
      SELECT 1
      FROM mimiciv_icu.chartevents ce
      CROSS JOIN params p
      WHERE ce.stay_id = idx.index_stay_id
        AND ce.charttime >= idx.index_icu_intime - p.win
        AND ce.charttime <= idx.index_icu_intime + p.win
        AND ce.value IS NOT NULL
        AND (
          -- O2 device signals invasive airway
          (ce.itemid IN (SELECT itemid FROM o2_device_itemids)
           AND ce.value IN ('Endotracheal tube', 'Tracheostomy tube'))
          OR
          -- Ventilator modes that imply invasive ventilation (common set from MIMIC Code concept)
          (ce.itemid IN (SELECT itemid FROM vent_mode_itemids)
           AND ce.value IN (
             '(S) CMV','APRV','Apnea Ventilation','CMV','CMV/ASSIST','CMV/AutoFlow',
             'CPAP/PPS','CPAP/PSV','MMV','P-CMV','PCV+','PRES/AC','PRVC/AC','PRVC/SIMV',
             'SIMV','SIMV/AutoFlow','SIMV/PRES','SIMV/PSV','SIMV/VOL','VOL/AC','PSV/SBT',
             'Ambient','ASV','VS','APV (cmv)','APV (simv)','P-SIMV'
           ))
        )
    ) THEN 1 ELSE 0 END AS mech_vent_within_6h,

    /* ----------------------------
       3) RRT/CRRT within window
          Based on:
          - procedureevents dialysis itemids
          - chartevents dialysis-related itemids
          - inputevents CRRT solutions
       ---------------------------- */
    CASE WHEN EXISTS (
      SELECT 1
      FROM mimiciv_icu.procedureevents pe
      CROSS JOIN params p
      WHERE pe.stay_id = idx.index_stay_id
        AND pe.starttime >= idx.index_icu_intime - p.win
        AND pe.starttime <= idx.index_icu_intime + p.win
        AND pe.itemid IN (
          225441, -- Hemodialysis
          225802, -- Dialysis - CRRT
          225803, -- Dialysis - CVVHD
          225809, -- Dialysis - CVVHDF
          225955, -- Dialysis - SCUF
          225805, -- Peritoneal Dialysis
          225436  -- CRRT Filter Change
        )
        AND pe.value IS NOT NULL
    )
    OR EXISTS (
      SELECT 1
      FROM mimiciv_icu.chartevents ce
      CROSS JOIN params p
      WHERE ce.stay_id = idx.index_stay_id
        AND ce.charttime >= idx.index_icu_intime - p.win
        AND ce.charttime <= idx.index_icu_intime + p.win
        AND ce.value IS NOT NULL
        AND ce.itemid IN (
          226118, 227357, 225725, -- dialysis catheter-related checkboxes
          226499, 224154, 224191, 226457, -- dialysis outputs/rates
          228004, 228005, 228006, -- citrate / replacement rates (CRRT)
          224144, 224145, 224149, 224150, 224151, 224152, 224153, -- pressures/flows
          224135, 224139, 224146, 225323, 225740, 225776, 225951, 225952, 225953, 225954,
          225956, 225958, 225961, 225963, 225965, 225976, 225977, 227124, 227290,
          227638, 227640, 227753
        )
    )
    OR EXISTS (
      SELECT 1
      FROM mimiciv_icu.inputevents ie
      CROSS JOIN params p
      WHERE ie.stay_id = idx.index_stay_id
        AND ie.starttime >= idx.index_icu_intime - p.win
        AND ie.starttime <= idx.index_icu_intime + p.win
        AND ie.itemid IN (
          227536, -- KCl (CRRT)
          227525  -- Calcium Gluconate (CRRT)
        )
        AND COALESCE(ie.amount, 0) > 0
    )
    THEN 1 ELSE 0 END AS rrt_within_6h,

    -- CRRT-specific (subflag)
    CASE WHEN EXISTS (
      SELECT 1
      FROM mimiciv_icu.procedureevents pe
      CROSS JOIN params p
      WHERE pe.stay_id = idx.index_stay_id
        AND pe.starttime >= idx.index_icu_intime - p.win
        AND pe.starttime <= idx.index_icu_intime + p.win
        AND pe.itemid IN (225802, 225803, 225809, 225955, 225436)
        AND pe.value IS NOT NULL
    )
    OR EXISTS (
      SELECT 1
      FROM mimiciv_icu.chartevents ce
      CROSS JOIN params p
      WHERE ce.stay_id = idx.index_stay_id
        AND ce.charttime >= idx.index_icu_intime - p.win
        AND ce.charttime <= idx.index_icu_intime + p.win
        AND ce.value IS NOT NULL
        AND ce.itemid IN (227290, 228004, 228005, 228006)
    )
    OR EXISTS (
      SELECT 1
      FROM mimiciv_icu.inputevents ie
      CROSS JOIN params p
      WHERE ie.stay_id = idx.index_stay_id
        AND ie.starttime >= idx.index_icu_intime - p.win
        AND ie.starttime <= idx.index_icu_intime + p.win
        AND ie.itemid IN (227536, 227525)
        AND COALESCE(ie.amount, 0) > 0
    )
    THEN 1 ELSE 0 END AS crrt_within_6h

  FROM idx
)
SELECT * FROM final;