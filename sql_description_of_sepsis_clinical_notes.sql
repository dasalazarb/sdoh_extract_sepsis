-- SELECT DISTINCT
--   P.subject_id
-- FROM mimiciv_hosp.patients P

-- SELECT DISTINCT
--   p.subject_id
-- FROM mimiciv_icu.icustays p

-- SELECT DISTINCT
-- 	p.subject_id
-- FROM mimiciv_derived.sepsis3 p

-- SELECT *
-- FROM sepsis_full
-- WHERE rn = 1
--   AND death_30d = 0
--   AND dod IS NOT NULL
--   AND dod <= suspected_infection_time + INTERVAL '30 days';

-- SELECT
--   gender,
--   COUNT(*) AS total_pacientes,
--   COUNT(*) FILTER (WHERE death_30d = 1) AS Died_in_30d_time_period,
--   COUNT(*) FILTER (WHERE death_30d = 0 AND dod IS NULL) AS censored_alive_patients,
--   COUNT(*) FILTER (WHERE death_30d = 0 AND dod IS NOT NULL AND dod > suspected_infection_time + INTERVAL '30 days') AS discharged_alive
-- FROM sepsis_full
-- WHERE rn = 1 AND dead_before_suspected = 0
-- GROUP BY gender
-- ORDER BY gender;

-- SELECT
--     s.subject_id,
--     COUNT(*) AS num_notes
-- FROM sepsis_full s
-- JOIN mimiciv_note.discharge n
--   ON s.subject_id = n.subject_id
-- WHERE s.rn = 1   -- para quedarte con la primera estancia de sepsis por paciente
-- GROUP BY s.subject_id
-- ORDER BY num_notes DESC;

-- SELECT 
--     subject_id,
--     hadm_id,
--     charttime,
--     text
-- FROM mimiciv_note.discharge
-- WHERE subject_id = 12468016   -- reemplaza con el ID del paciente
-- ORDER BY charttime;

-- SELECT *
-- FROM mimiciv_note.discharge
-- LIMIT 10;

-- SELECT 
--     n.subject_id,
--     n.hadm_id,
--     n.charttime,
--     n.text
-- FROM mimiciv_note.discharge n
-- JOIN sepsis_full s
--   ON n.subject_id = s.subject_id
-- JOIN mimiciv_icu.icustays i
--   ON s.stay_id = i.stay_id
-- WHERE s.rn = 1
--   AND DATE(n.charttime) >= DATE(i.intime) - INTERVAL '1 year'   -- un año antes de ingreso a UCI
--   AND DATE(n.charttime) <= DATE(s.suspected_infection_time)     -- hasta la sospecha
-- ORDER BY n.charttime;

-- Number of clilnical notes
-- SELECT 
--     s.subject_id,
--     COUNT(*) AS num_notes
-- FROM mimiciv_note.discharge n
-- JOIN sepsis_full s
--   ON n.subject_id = s.subject_id
-- WHERE s.rn = 1
--   AND DATE(n.charttime) >= DATE(s.suspected_infection_time) - INTERVAL '1 year'
--   AND DATE(n.charttime) <= DATE(s.suspected_infection_time)
-- GROUP BY s.subject_id
-- ORDER BY num_notes DESC;

--  length of clinical notes: characters and words
SELECT 
    n.subject_id,
    COUNT(*) AS num_notes,

    -- 📌 Totales
    -- SUM(LENGTH(n.text)) AS total_chars,
    SUM(LENGTH(regexp_replace(n.text, '[^A-Za-zÁÉÍÓÚáéíóúÑñ]', '', 'g'))) AS total_letters,
    SUM(array_length(regexp_split_to_array(n.text, '\s+'), 1)) AS total_words,

    -- 📌 Promedios por nota
    -- AVG(LENGTH(n.text)) AS avg_chars_per_note,
    AVG(LENGTH(regexp_replace(n.text, '[^A-Za-zÁÉÍÓÚáéíóúÑñ]', '', 'g'))) AS avg_letters_per_note,
    AVG(array_length(regexp_split_to_array(n.text, '\s+'), 1)) AS avg_words_per_note,

    -- 📌 Desviación estándar por nota
    STDDEV(LENGTH(regexp_replace(n.text, '[^A-Za-zÁÉÍÓÚáéíóúÑñ]', '', 'g'))) AS stddev_chars,
    STDDEV(array_length(regexp_split_to_array(n.text, '\s+'), 1)) AS stddev_words,
	
    -- 📌 Mínimos y máximos
    MIN(LENGTH(regexp_replace(n.text, '[^A-Za-zÁÉÍÓÚáéíóúÑñ]', '', 'g'))) AS min_letters_per_note,
    MAX(LENGTH(regexp_replace(n.text, '[^A-Za-zÁÉÍÓÚáéíóúÑñ]', '', 'g'))) AS max_letters_per_note,
    MIN(array_length(regexp_split_to_array(n.text, '\s+'), 1)) AS min_words,
    MAX(array_length(regexp_split_to_array(n.text, '\s+'), 1)) AS max_words

FROM mimiciv_note.discharge n
JOIN sepsis_full s
  ON n.subject_id = s.subject_id
JOIN mimiciv_icu.icustays i
  ON s.stay_id = i.stay_id
WHERE s.rn = 1
  AND DATE(n.charttime) >= DATE(i.intime) - INTERVAL '1 year'
  AND DATE(n.charttime) <= DATE(s.suspected_infection_time)
GROUP BY n.subject_id
ORDER BY num_notes DESC;
