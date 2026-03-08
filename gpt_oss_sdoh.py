import torch
import json
import pandas as pd
import utils_llm as ul
from tqdm import tqdm
from datetime import datetime
#import random
import logging
#from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)
from functools import wraps
import time

# ================== CONFIGURACIÓN LOGGING ==================
timestamp_log = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"run_log_gpt_oss_{timestamp_log}.log"

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
    model_id = '/data/salazarda/data/models/gpt-oss-21b'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    pad_token_id=tokenizer.eos_token_id, device=0)
    return model_id, tokenizer, model, pipe


# ================== BLOQUE 3 ==================
@log_time
def bloque3(model_id, tokenizer, model):
    # === Setup modelo GPT-OSS ===
    def query_gptoss(prompts, max_new_tokens=10000):
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

    # === Código original ===
    time_start = datetime.now()
    timestamp = time_start.strftime("%Y%m%d_%H%M%S")
    
    subject_and_hadm_ids = pd.read_csv(
        # '/data/salazarda/data/sdoh/SDOH_MIMICIII_physio_release.csv'
        # '/data/salazarda/data/sdoh/temp_notes_filtered_ids_more_than_one_cc.csv'
        # '/data/salazarda/data/sdoh/archivos_sin_texto_primeras_20_palabras.csv'
        '/data/salazarda/data/sdoh/outputs/PEND_keys_gpt-oss_no_en_all_notes.csv'
    )
    # subject_and_hadm_ids = pd.read_csv('C:/Users/salazarda/Downloads/SDOH_MIMICIII_physio_release.csv')
    # subject_and_hadm_ids = list(subject_and_hadm_ids.loc[:,['patient_id', 'note_id']].drop_duplicates().itertuples(index=False, name=None))
    subject_and_hadm_ids = list(subject_and_hadm_ids.loc[:,['subject_id', 'hadm_id', 'note_id']].drop_duplicates().itertuples(index=False, name=None))
    subject_and_hadm_ids = subject_and_hadm_ids[1600:]  # DEBUG opcional

    # notes = ul.get_clinical_notes_mimic3(subject_and_hadm_ids)
    # notes = ul.get_clinical_notes_mimic4(subject_and_hadm_ids)
    notes = ul.get_clinical_notes_mimic4_from_csv(subject_and_hadm_ids, base_dir="/data/salazarda/data/sdoh/notes_by_patient_gptoss_2048tokens")


    # notes = notes[0:2]  # DEBUG opcional

    prompts, metadata = [], []
    for subject_id, hadm_id, row_id, charttime, note_text in notes:
        prompts.append(ul.sdh_prompt_guevara_v3(note_text))
        metadata.append({
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "row_id": row_id,
            "charttime": charttime.isoformat() if charttime else None
        })

    batch_size = 4  # 2–4 como máximo para 20B/120B
    final_outputs = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_meta = metadata[i:i + batch_size]
        batch_responses = query_gptoss(batch_prompts, max_new_tokens=50000)

        for meta, raw, prompt in zip(batch_meta, batch_responses, batch_prompts):
            # limpiar: quitar el prompt del output si el modelo lo repite
            text = raw.replace(prompt, "").strip()
            text = {'text': text}
            final_outputs.append({**meta, **text})

    ul.save_to_jsonl(final_outputs, model_id, timestamp)

    n = len(list(set([i['subject_id'] for i in final_outputs])))
    logging.info(
        f'For {n} patients and {len(notes)} notes, it took {datetime.now() - time_start}'
    )


# ================== MAIN ==================
def main():
    bloque1()
    model_id, tokenizer, model, pipe = bloque2()
    bloque3(model_id, tokenizer, model)


if __name__ == "__main__":
    main()
