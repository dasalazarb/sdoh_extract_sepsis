import torch
import json
import pandas as pd
import utils_llm as ul
from tqdm import tqdm
from datetime import datetime
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from functools import wraps
import time


# ================== CONFIGURACIÓN LOGGING ==================
timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"run_log_deepseek_{timestamp_log}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def log_time(func):
    """Decorador para medir y registrar el tiempo de ejecución de cada bloque."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logging.info(f"Inicio: {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"Fin: {func.__name__} - Tiempo: {elapsed:.2f} segundos")
        return result
    return wrapper


# ================== BLOQUE 1 ==================
@log_time
def bloque1():
    logging.info("BLOQUE 1 cargado (imports y setup inicial).")


# ================== BLOQUE 2 ==================
@log_time
def bloque2():
    model_id = '/data/salazarda/data/models/DeepSeek-R1-Distill-Qwen-14B'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    pad_token_id=tokenizer.eos_token_id, device=0)
    return model_id, tokenizer, model, pipe


# ================== BLOQUE 3 ==================
@log_time
def bloque3(model_id, tokenizer, model, pipe):
    time_start = datetime.now()
    timestamp = time_start.strftime("%Y%m%d_%H%M%S")

    # === Cargar dataset ===
    subject_and_hadm_ids = pd.read_csv(
        # '/data/salazarda/data/sdoh/SDOH_MIMICIII_physio_release.csv'
        # '/data/salazarda/data/sdoh/temp_notes_filtered_ids.csv'
        '/data/salazarda/data/sdoh/archivos_sin_texto_primeras_20_palabras.csv'
    )
    # subject_and_hadm_ids = pd.read_csv('C:/Users/salazarda/Downloads/SDOH_MIMICIII_physio_release.csv')
    # subject_and_hadm_ids = list(subject_and_hadm_ids.loc[:,['patient_id', 'note_id']].drop_duplicates().itertuples(index=False, name=None))
    subject_and_hadm_ids = list(subject_and_hadm_ids.loc[:,['subject_id', 'hadm_id', 'note_id']].drop_duplicates().itertuples(index=False, name=None))
    subject_and_hadm_ids = subject_and_hadm_ids[4480:4550]  # DEBUG opcional
    
    # === Obtener notas clínicas ===
    # notes = ul.get_clinical_notes_mimic3(subject_and_hadm_ids)
    # notes = ul.get_clinical_notes_mimic4(subject_and_hadm_ids)
    notes = ul.get_clinical_notes_mimic4_from_csv(subject_and_hadm_ids, base_dir="/data/salazarda/data/sdoh/notes_by_patient_deepseek_2048tokens")
    
    # notes = notes[0:2]  # DEBUG opcional

    # === Construcción de prompts y metadata ===
    prompts, metadata = [], []
    for subject_id, hadm_id, row_id, charttime, note_text in notes:
        prompts.append(ul.sdh_prompt_guevara_v3(note_text))
        metadata.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "row_id": row_id,
            "charttime": charttime.isoformat() if charttime else None
        })

    # === Procesar en lotes ===
    batch_size = 8  # 2–4 como máximo para 20B/120B
    final_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_meta = metadata[i:i+batch_size]
        batch_responses = pipe(batch_prompts, 
                               max_new_tokens=500)
        
        for meta, raw, prompt in zip(batch_meta, batch_responses, batch_prompts):
            text = raw[0]['generated_text']
            if prompt in text:
                text = text.replace(prompt, "").strip()
            text = {'text': text}
            final_outputs.append({**meta, **text})

    # === Guardar resultados ===
    ul.save_to_jsonl(final_outputs, model_id, timestamp)

    n = len(list(set([i['subject_id'] for i in final_outputs])))
    logging.info(
        f'For {n} patients and {len(notes)} notes, it took {datetime.now() - time_start}'
    )


# ================== MAIN ==================
def main():
    bloque1()
    model_id, tokenizer, model, pipe = bloque2()
    bloque3(model_id, tokenizer, model, pipe)


if __name__ == "__main__":
    main()
