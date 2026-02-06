import psycopg2
import pandas as pd
import re
import json
from datetime import datetime

def get_clinical_notes_mimic3(pairs):
    conn = psycopg2.connect(
        dbname="mimic3",
        user="postgres",
        password="aasalazarda",
        host="localhost",
        port="5433"
    )
    cur = conn.cursor()
    query = """
        SELECT n.subject_id, n.hadm_id, n.row_id, n.charttime, n.text
          FROM mimiciii.noteevents AS n
         WHERE (n.subject_id, n.row_id) IN %s;
    """
    cur.execute(query, (tuple(pairs),))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # return [
    #     {"subject_id": sid, "hadm_id": hid, "row_id": rid, "charttime": ct, "text": txt}
    #     for sid, hid, rid, ct, txt in rows
    # ]
    return rows

def get_clinical_notes_mimic4(pairs):
    """
    pairs = list of (subject_id, note_id)
    """

    conn = psycopg2.connect(
        dbname="mimic4",
        #user="postgres",
        #password="aasalazarda",
        host="localhost",
        port="5434"
    )
    cur = conn.cursor()
    
    query = """
        SELECT n.subject_id,
               n.hadm_id,
               n.note_id,
               n.charttime,
               n.text
          FROM mimiciv_note.discharge AS n
         WHERE (n.subject_id, n.hadm_id) IN %s;
    """
    
    cur.execute(query, (tuple(pairs),))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    return rows

import os

def get_clinical_notes_mimic4_from_csv(pairs,
                              base_dir="/data/salazarda/data/sdoh/notes_by_patient"):
    """
    Carga las clinical notes de MIMIC-IV desde archivos ya procesados.

    pairs: lista de tuplas (subject_id, hadm_id, note_id)
           por ejemplo: [(10000123, 2000456, 3001), (10000123, 2000789, 3002), ...]

    base_dir: carpeta donde guardaste los {subject_id}_{hadm_id}_{note_id}.txt

    Devuelve: lista de tuplas (subject_id, hadm_id, note_id, charttime, text)
              aquí note_id y charttime se ponen en None, para mantener la firma
              similar a la versión SQL anterior.
    """
    results = []

    for subject_id, hadm_id, note_id in pairs:
        fname = f"{subject_id}_{hadm_id}_{note_id}.txt"
        fpath = os.path.join(base_dir, fname)

        if not os.path.isfile(fpath):
            # Si no existe el archivo, lo saltamos o puedes hacer logging
            # print(f"[WARN] No se encontró archivo para {subject_id}, {hadm_id}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        # Mantenemos la estructura (sid, hadm, note_id, charttime, text)
        # charttime = None porque no los tenemos aquí
        results.append((subject_id, hadm_id, note_id, None, text))

    return results


def get_notes_for_first_n_patients(n=10):
    conn = psycopg2.connect(
        dbname="mimic4",
        user="postgres",
        password="aasalazarda",
        host="localhost",
        port="5433"
    )
    cur = conn.cursor()
    query = f"""
        SELECT n.subject_id, n.hadm_id, n.charttime, n.text
        FROM mimiciv_note.discharge n
        JOIN (
            SELECT DISTINCT subject_id 
            FROM mimiciv_note.discharge
            ORDER BY subject_id ASC
            LIMIT %s
        ) p ON n.subject_id = p.subject_id
    """
    cur.execute(query, (n,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results

def get_notes_for_first_n_notes(n=10):
    conn = psycopg2.connect(
        dbname="mimic4",
        user="postgres",
        password="aasalazarda",
        host="localhost",
        port="5433"
    )
    cur = conn.cursor()
    query = f"""
        SELECT n.subject_id, n.hadm_id, n.charttime, n.text
        FROM mimiciv_note.discharge n
        WHERE n.hadm_id IN (
            SELECT hadm_id
            FROM mimiciv_note.discharge
            ORDER BY subject_id ASC
            LIMIT %s
        )
    """
    cur.execute(query, (n,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return results
    
    
def get_patients(limit=10):
    conn = psycopg2.connect(
        dbname="mimic4",
        user="postgres",
        password="aasalazarda",
        host="localhost",
        port="5433"
    )
    cur = conn.cursor()
    query = f"""
        SELECT subject_id
        FROM mimiciv_hosp.patients
        ORDER BY RANDOM()
        LIMIT %s;
    """
    cur.execute(query, (limit,))
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [row[0] for row in results]
    
def parse_sdh_response(text):
    output = {}
    lines = text.strip().splitlines()

    for line in lines:
        if ":" not in line:
            continue

        match = re.match(r"^SDH_(.*?):\s*(Yes|No|Not)\s*[-:]?\s*\[?(.*?)]?$", line.strip())
        if match:
            category, present, evidence = match.groups()
            category = category.strip()
            evidence = evidence.strip()
            if not evidence or evidence.lower() in ["n/a", "none", "not mentioned", "na", ""]:
                evidence = "There is no evidence"
            output[category] = {"present": present, "evidence": evidence}
    
    return output

def analyze_sdh_for_subject(subject_id):
    note = get_clinical_note(subject_id)
    if not note:
        return {"subject_id": subject_id, "error": "No note found"}
    
    prompt = sdh_prompt(note)
    response = pipe(prompt, max_new_tokens=400)[0]['generated_text']
    parsed = parse_sdh_response(response)
    
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "charttime": charttime.isoformat() if charttime else None,
        **parsed
    }

def process_note_with_sdh_extraction(subject_id, hadm_id, charttime, note_text, pipe):
    prompt = sdh_prompt(note_text)
    response = pipe(prompt, max_new_tokens=400)[0]['generated_text']
    if prompt in response:
        parsed = response.replace(prompt, '').strip()
    else:
        parsed = respopnse.strip()
    # parsed = parse_sdh_response(response)
    
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "charttime": charttime.isoformat() if charttime else None,
        **parsed
    }

def sdh_prompt(note_text):
    # 5. **Relationship status**: Whether the patient is married, divorced, single, has a partner, or is widowed.
    # SDH_Relationship status: [Yes/No] - [Give a short evidence sentence]
    # 7. **Substance Use**: Any mention of alcohol, drug, or tobacco use, including current use, past use, or explicit denial of use.
    # SDH_Substance Use: [Yes/No] - [Give a short evidence sentence]
    return f"""
    Analyze the following clinical note and indicate whether each of the following seven social determinants of health (SDH) is specifically mentioned:
        
    1. **Employment status**: Whether the patient is currently employed, unemployed, retired, on disability, or has a job title or income source.
    2. **Housing issues**: Any mention of homelessness, unstable housing, living in shelters, or housing concerns (e.g., can't afford rent, frequent moves).
    3. **Transportation issues**: Any reference to transportation difficulties, lack of car access, reliance on public transit, missed appointments due to transportation.
    4. **Parental status**: Whether the patient has children or dependents, or is a caregiver to minors.
    5. **Social support**: It does include informal or emotional support from family members, friends, or romantic partners unless such support is clearly mediated through a formal care plan by a social worker or case manager.
    
    Answer with **"Yes" or "No"** for each item, and include a **short evidence sentence** for each item. If not mentioned, say: *There is no evidence.*
    
    Complete this format:
    
    SDH_Employment status_: [Yes/No] - [Give a short evidence sentence]
    SDH_Housing issues: [Yes/No] - [Give a short evidence sentence]
    SDH_Transportation issues: [Yes/No] - [Give a short evidence sentence]
    SDH_Parental status: [Yes/No] - [Give a short evidence sentence]
    SDH_Social support: [Yes/No] - [Give a short evidence sentence]
    
    ---
    
    Now analyze the following clinical note:
    
    \"\"\"
    {note_text}
    \"\"\"
    

    """

def sdh_prompt_guevara(note_text):
    return f"""
    You will be provided with the following information:
    1. An arbitrary text sample. The sample is delimited with triple backticks.
    2.  List of categories the text sample can be assigned to. The list is delimited with square brackets. The categories in the list are enclosed in the single quotes and comma separated.
    3. The definition that help you guide to identify the specific label for each category.
    4.  Examples of text samples and their assigned categories. The examples are delimited with triple backticks. The assigned categories are enclosed in a list-like structure. These examples are to be used as training data
    
    Perform the following tasks:
    1. Identify to which category the provided text belongs to with the highest probability.
    2. Assign the provided text to that category.
    3. Provide your response in a JSON format containing a single key `label` and a value corresponding to the assigned category. Do not provide any additional information except the JSON.

    List of categories:
    1. Employment status: [EMPLOYMENT, unknown]
    2. Housing issues: [HOUSING, unknown]
    3. Transportation issues: [TRANSPORTATION, unknown]
    4. Parental status: [PARENT, unknown]
    5. Relationship status: [RELATIONSHIP, unknown]
    6. Social support: [SOCIAL_SUPPORT, unknown]

    Text sample: ```Childcare provider offers after-school tutoring services helping child stay on track academically while parent undergoes treatment```
    Your JSON response:
      "Employment status": "unknown",
      "Housing issues": "unknown",
      "Transportation issues": "unknown",
      "Parental status": "PARENT",
      "Relationship status": "unknown",
      "Social support": "unknown"
    
    Clinical note:
    
    \"\"\"
    {note_text}
    \"\"\"

    """

def sdh_prompt_amrutha(note_text):
    return f"""
    Your are a NLP expert assistant. You will be provided with a prompt which consists of three sections: "Input information", "Instructions" and "Clinical note".

    ## Section 1: Input information
    You will be provided with the following information:
    1. An arbitrary text sample (CLINICAL_NOTE). The sample is delimited with triple backticks.
    2. List of categories that we want to search in the text.
    3. The definition that help you guide to identify the specific label for each category.
    4. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. These examples are to be used as training data.

    ## Section 2: Instructions (follow the steps in order)
    General: Your task is to extract structured information about Social Determinants of Health (SDOH) from the free-text clinical note provided in the "Input" section.
    1. Analyse the text and determine whether any of the following six Social Determinants of Health (SDOH) are explicitly stated based on the definitions below.  
    2. For each SDOH, select exactly one label from the bracketed list.  
       • If the note contains none of the listed concepts for that SDOH, choose “unknown”.  
       • Do NOT invent new labels, combine labels, or add explanations.
       • Provide a brief evidence phrase from the note that supports your classification.
    3. Return your answer only as a JSON object with six keys (one per SDOH) and values equal to the chosen label.

    ### Categories and Attributes:
    List of SDOH categories and their definitions:
    1. EMPLOYMENT: 
        LABELS: [EMPLOYMENT_employed, EMPLOYMENT_unemployed, EMPLOYMENT_underemployed, EMPLOYMENT_disability, EMPLOYMENT_retired, EMPLOYMENT_student, EMPLOYMENT_unknown]
        The attribute EMPLOYMENT_unknown means the CLINICAL_NOTE does not mention information regarding the current employment status of the patient at the time of <<text>>.
        The attribute EMPLOYMENT_employed means the CLINICAL_NOTE states that the patient is currently employed.
        The attribute EMPLOYMENT_unemployed means the CLINICAL_NOTE states that the patient is currently unemployed, e.g., out of work, jobless, laid-off, etc.
        The attribute EMPLOYMENT_underemployed means the CLINICAL_NOTE states that the patient is currently underemployed, e.g., low-paying job, part-time worker etc.
        The attribute EMPLOYMENT_disability means the CLINICAL_NOTE states that the patient is currently not working due to a disability, e.g., unable to work because of a physical or mental health condition, receiving disability benefits, or described as disabled or incapacitated.
        The attribute EMPLOYMENT_retired means the CLINICAL_NOTE states that the patient is currently retired, e.g., inactive, retired person, pensioner, etc.
        The attribute EMPLOYMENT_student means the CLINICAL_NOTE states that the patient is currently a student, e.g. goes to college/university, graduate/undergraduate student, etc.

    2. HOUSING:
        LABELS: [HOUSING_financial_status, HOUSING_undomiciled, HOUSING_other, HOUSING_unknown]
        The attribute HOUSING_financial_status means the CLINICAL_NOTE states that the patient's housing situation is affected by their financial status, e.g., unable to pay rent, facing eviction due to financial hardship, etc.
        The attribute HOUSING_undomiciled means the CLINICAL_NOTE states that the patient is currently without a fixed dwelling, e.g., homeless, living in a shelter, etc.
        The attribute HOUSING_other means the CLINICAL_NOTE states that the patient's housing situation does not clearly fall under the categories of financial status or undomiciled, but still provides information about housing, e.g., living in transitional housing, incarceration, or other arrangements not otherwise specified, etc.
        The attribute HOUSING_unknown means the CLINICAL_NOTE does not mention information regarding the patient's housing situation.

    3. TRANSPORTATION:
        LABELS: [TRANSPORTATION_distance, TRANSPORTATION_resources, TRANSPORTATION_other, TRANSPORTATION_unknown]
        The attribute TRANSPORTATION_Distance means the CLINICAL_NOTE states that the patient has transportation difficulties specifically related to long travel distances, e.g., the clinic is far from home, requires extended travel time, or is located in a different city or rural area with limited access, etc.
        The attribute TRANSPORTATION_Resources means the CLINICAL_NOTE states that the patient has transportation difficulties related to limited access to transportation resources, e.g., lack of a personal vehicle, inability to afford public transit, missed appointments due to transportation issues, or reliance on others for transportation, etc.
        The attribute TRANSPORTATION_other means the CLINICAL_NOTE states that the patient has transportation difficulties related to other factors.
        The attribute TRANSPORTATION_unknown means the CLINICAL_NOTE does not mention information regarding the patient's transportation situation.

    4. PARENTAL: 
        LABELS: [PARENTAL_yes, PARENTAL_no, PARENTAL_unknown]
        The attribute PARENTAL_yes means the CLINICAL_NOTE states that the patient has a child under 18 years old.
        The attribute PARENTAL_no means the CLINICAL_NOTE states that the patient does not have a child under 18 years old.
        The attribute PARENTAL_unknown means the CLINICAL_NOTE does not mention information regarding the patient's parental status.

    5. RELATIONSHIP:
        LABELS: [RELATIONSHIP_single, RELATIONSHIP_married, RELATIONSHIP_divorced, RELATIONSHIP_widowed, RELATIONSHIP_partnered, RELATIONSHIP_unknown]
        The attribute RELATIONSHIP_single means the CLINICAL_NOTE states that the patient is single.
        The attribute RELATIONSHIP_married means the CLINICAL_NOTE states that the patient is married.
        The attribute RELATIONSHIP_divorced means the CLINICAL_NOTE states that the patient is divorced.
        The attribute RELATIONSHIP_widowed means the CLINICAL_NOTE states that the patient is widowed.
        The attribute RELATIONSHIP_partnered means the CLINICAL_NOTE states that the patient is partnered.
        The attribute RELATIONSHIP_unknown means the CLINICAL_NOTE does not mention information regarding the patient's relationship status.

    6. SOCIAL:
        LABELS: [SOCIAL_presence, SOCIAL_absence, SOCIAL_unknown]
        The attribute SOCIAL_presence means the CLINICAL_NOTE states that the patient has informal or emotional support from family members, friends, or romantic partners.
        The attribute SOCIAL_absence means the CLINICAL_NOTE states that the patient does not have informal or emotional support from family members, friends, or romantic partners.
        The attribute SOCIAL_unknown means the CLINICAL_NOTE does not mention information regarding the patient's informal or emotional support.

    OUTPUT TEMPLATE (use exactly this format)

      "EMPLOYMENT": "…", "EMPLOYMENT_evidence": "…",
      "HOUSING": "…", "HOUSING_evidence": "…",
      "TRANSPORTATION": "…", "TRANSPORTATION_evidence": "…",
      "PARENTAL": "…", "PARENTAL_evidence": "…",
      "RELATIONSHIP": "…", "RELATIONSHIP_evidence": "…",
      "SOCIAL": "…", "SOCIAL_evidence": "…"

    Text sample example: ```38 y/o F, single mother of 2 (4 y/o and 6 y/o), h/o HTN and anxiety presents with medication nonadherence due to unstable PT barista job (~20 h/wk) and unreliable public transport causing missed appts and work shifts. Reports 2 mo rent arrears, late-payment notice pending eviction. No personal vehicle, bus cuts limit mobility. Ex-spouse provides no support; sister OOS; one friend for occasional childcare. Limited social support. Plan: continue lisinopril 10 mg daily, add SSRI; refer to housing assistance, workforce development, bus-pass voucher, subsidized childcare, social work, and food pantry.
    ```
    
    YOUR JSON RESPONSE:
    
        "EMPLOYMENT": "underemployed", 
        "EMPLOYMENT_evidence": "unstable PT barista job (~20 h/wk)",

        "HOUSING": "instability", 
        "HOUSING_evidence": "2 mo rent arrears, late-payment notice pending eviction", 

        "TRANSPORTATION": "resources", 
        "TRANSPORTATION_evidence": "unreliable public transport causing missed appts and work shifts; no personal vehicle, bus cuts limit mobility", 

        "PARENTAL": "yes", 
        "PARENTAL_evidence": "single mother of 2 (4 y/o and 6 y/o)", 

        "RELATIONSHIP": "divorced", 
        "RELATIONSHIP_evidence": "Ex-spouse provides no support", 

        "SOCIAL": "absence", 
        "SOCIAL_evidence": "Limited social support; sister OOS; one friend for occasional childcare", 
    ---

    ## Section 3: Clinical note
    Now analyze the following CLINICAL_NOTE:
    
    \"\"\"
    {note_text}
    \"\"\"
    """

def sdh_prompt_guevara_v2(note_text):
    return f"""
    Your are a NLP expert assistant. You will be provided with the following information:
    1. An arbitrary text sample. The sample is delimited with triple backticks.
    2. List of categories that we want to search in the text.
    3. The definition that help you guide to identify the specific label for each category.
    4. Examples of text samples and their assigned categories. The examples are delimited with triple backticks. These examples are to be used as training data.
    
    TASK (follow the steps in order)
    1. Analyse the text and determine whether any of the following six Social Determinants of Health (SDOH) are explicitly stated based on the definitions below.  
    2. For each SDOH, select exactly one label from the bracketed list.  
       • If the note contains none of the listed concepts for that SDOH, choose “unknown”.  
       • Do NOT invent new labels, combine labels, or add explanations.  
    3. Return your answer only as a JSON object with six keys (one per SDOH) and values equal to the chosen label.  
       - Do not include any additional text or commentary.
       - Do not include your rational.
        
    List of SDOH categories and their definitions:
    1. Employment status: whether the patient is currently employed, unemployed (e.g., out of work, jobless, laid-off), underemployed (e.g., part-time work, low pay), has a disability that prevents working (e.g., physical or mental health condition, receiving disability benefits), retired (e.g., inactive, pensioner), or is a student (e.g., enrolled in school, college, or university). Unknown if the clinical note does not mention the patient’s employment status.
    LABELS: [employed, unemployed, underemployed, disability, retired, student, unknown]
    2. Housing issues: whether the patient’s housing situation is affected by financial hardship (e.g., unable to pay rent, facing eviction), is undomiciled (e.g., homeless, living in a shelter), or falls under another category (e.g., transitional housing, incarceration, other non-standard arrangements). Unknown if the clinical note does not mention the patient’s housing situation.
    LABELS: [financial_status, undomiciled, other, unknown]
    3. Transportation issues: whether the patient experiences transportation difficulties related to distance (e.g., clinic far from home, long travel time, rural area), resources (e.g., no personal vehicle, inability to afford public transit, reliance on others), or other factors not covered by distance or resources. Unknown if the clinical note does not mention transportation.
    LABELS: [distance, resources, other, unknown]
    4. Parental status: whether the patient has at least one child under 18 years old, does not have a child under 18 years old, or if this is not mentioned in the clinical note.
    LABELS: [yes, no, unknown]
    5. Relationship status: whether the patient is single, married, divorced, widowed, or partnered, as stated in the clinical note. Unknown if relationship status is not mentioned.
    LABELS: [single, married, divorced, widowed, partnered, unknown]
    6. Social support: whether the patient has (presence) or does not have (absence) informal or emotional support from family members, friends, or romantic partners. Unknown if the clinical note does not provide this information.
    LABELS: [presence, absence, unknown]

    OUTPUT TEMPLATE (use exactly this format)
    
      "Employment status": "…",
      "Housing issues": "…",
      "Transportation issues": "…",
      "Parental status": "…",
      "Relationship status": "…",
      "Social support": "…"
    
    Text sample: ```38 y/o F, single mother of 2 (4 y/o and 6 y/o), h/o HTN and anxiety presents with medication nonadherence due to unstable PT barista job (~20 h/wk) and unreliable public transport causing missed appts and work shifts. Reports 2 mo rent arrears, late-payment notice pending eviction. No personal vehicle, bus cuts limit mobility. Ex-spouse provides no support; sister OOS; one friend for occasional childcare. Limited social support. Plan: continue lisinopril 10 mg daily, add SSRI; refer to housing assistance, workforce development, bus-pass voucher, subsidized childcare, social work, and food pantry.
    ```
    
    YOUR JSON RESPONSE:
    
      "Employment status": "employed",
      "Housing issues": "other",
      "Transportation issues": "other",
      "Parental status": "yes",
      "Relationship status": "divorced",
      "Social support": "absence"
    
    ---
    Now analyze the following clinical note:
    
    \"\"\"
    {note_text}
    \"\"\"
    """

def sdh_prompt_guevara_v3(note_text):
    return f"""
    You are an NLP expert assistant. Your task is to extract structured information about Social Determinants of Health (SDOH) from clinical notes. 
    
    INPUT INFORMATION:
    1. Clinical note text, delimited by triple backticks.
    2. Four SDOH categories with their definitions and labels.
    3. Examples of annotated notes with expected JSON outputs.
    
    TASK:
    1. Analyze the provided clinical note.
    2. For each of the four SDOH categories, select exactly one label from the provided list.
    3. Always return a valid JSON object following the schema below.
    4. Use only the listed labels. If the information is not present or different as the labels suggested, select "unknown".

    CATEGORIES & LABELS:
    1. Employment status: [employed, underemployed, unemployed, disability, retired, student, not_mentioned, unsure]
        - employed → patient has evidence of holding a regular job or steady work.
        - underemployed → patient has evidence of working but with insufficient hours, pay, or job stability.
        - unemployed → patient has evidence of not currently working and actively lacking employment.
        - disability → patient has evidence of being unable to work due to physical or mental disability.
        - retired → patient has evidence of having left the workforce due to retirement.
        - student → patient has evidence of being enrolled in school, college, or training as primary occupation.
        - not_mentioned → no information about employment status is provided.
        - unsure → ambiguous or unclear information regarding employment status.
    2. Parental status: if the patient has a child under 18 years old [yes, no, not_mentioned, unsure]
    3. Housing issues: [financial_status, undomiciled, other, not_mentioned, unsure]
        - financial_status → patient has evidence of housing difficulties related to financial strain, such as inability to afford rent or utilities.
        - undomiciled → patient has evidence of being homeless or lacking stable housing.
        - other → patient has evidence of housing problems not captured by the other categories.
        - not_mentioned → no information about housing issues is provided.
        - unsure → ambiguous or unclear information regarding housing issues.
    4. Transportation issues: [distance, resources, other, not_mentioned, unsure]
        - distance → patient has evidence of transportation difficulties due to long travel distances to healthcare or essential services.
        - resources → patient has evidence of lacking access to transportation resources, such as a car, money for fares, or reliable public transit.
        - other → patient has evidence of transportation problems not captured by the other categories.
        - not_mentioned → no information about transportation issues is provided.
        - unsure → ambiguous or unclear information regarding transportation issues.
    5. Relationship status: [married, partnered, divorced, widowed, single, not_mentioned, unsure]
        - married → patient has evidence of being legally married.
        - partnered → patient has evidence of being in a committed romantic relationship but not legally married.
        - divorced → patient has evidence of having legally ended a marriage.
        - widowed → patient has evidence of having lost a spouse through death.
        - single → patient has evidence of not being in a current romantic relationship and never married.
        - not_mentioned → no information about relationship status is provided.
        - unsure → ambiguous or unclear information regarding relationship status.
    6. Social support: [plus, minus, not_mentioned, unsure]
        - plus → patient has evidence of positive support from family/friends/community.
        - minus → patient has evidence of family/friends but support is negative or absent.
        - not_mentioned → no information about social support is provided.
        - unsure → ambiguous or unclear information regarding social support.

    OUTPUT SCHEMA (always valid JSON, no extra text):
    [
      "Employment status": "[label]",
      "Housing issues": "[label]",
      "Transportation issues": "[label]",
      "Parental status": "[label]",
      "Relationship status": "[label]",
      "Social support": "[label]"
    ]

    EXAMPLES:
    Example 1
    Text: ```38 y/o F, single mother of 2 (4 y/o and 6 y/o), h/o HTN and anxiety. Works irregular shifts as a PT barista (~20 h/wk). Reports difficulty keeping appointments due to unstable work schedule and unreliable transport. Behind on rent with eviction notice pending. States that ex-spouse is hostile and provides no help, sister lives out of state, and friends are "too busy" to assist. Patient feels abandoned and overwhelmed.```
    Output:
    [
      "Employment status": "employed",
      "Housing issues": "unknown",
      "Transportation issues": "distance",
      "Parental status": "yes",
      "Relationship status": "divorced",
      "Social support": "minus"
    ]

    Example 2 
    Text: ```Patient presents for routine physical. No mention of social history.```
    Output:
    [
      "Employment status": "unknown",
      "Housing issues": "unknown",
      "Transportation issues": "unknown",
      "Parental status": "unknown",
      "Relationship status": "unknown",
      "Social support": "unknown"
    ]

    Example 3 
    Text: ```19 y/o college student living in dorms, no children, partnered, parents provide emotional and financial support. No transport or housing issues reported.```
    Output:
    [
      "Employment status": "student",
      "Housing issues": "unknown",
      "Transportation issues": "unknown",
      "Parental status": "no",
      "Relationship status": "partnered",
      "Social support": "plus"
    ]

    Example 4 
    Text: ```72 y/o M, retired teacher, widowed, lives independently with strong support from adult children and church community. No housing or transport concerns.``` 
    Output:
    [
      "Employment status": "retired",
      "Housing issues": "unknown",
      "Transportation issues": "unknown",
      "Parental status": "unknown",
      "Relationship status": "widowed",
      "Social support": "plus"
    ]

    Example 5 
    Text: ```45 y/o M, employed full-time as mechanic, married. Reports ongoing conflict with spouse who is verbally abusive and unsupportive. Patient states that siblings refuse to help and have cut off communication. Feels isolated and reports that family interactions increase stress and worsen mood.```
    Output:
    [
      "Employment status": "employed",
      "Housing issues": "unknown",
      "Transportation issues": "unknown",
      "Parental status": "unknown",
      "Relationship status": "married",
      "Social support": "minus"
    ]

    ---
    Now analyze the following clinical note:

    ```{note_text}```
    """

def sdh_single_prompt(note_text: str, sdoh: str, sdoh_def: str) -> str:
    return f"""
    Your are a NLP expert assistant. You will be provided with the following information:
    1. An arbitrary text sample. The sample is delimited with triple backticks.
    2. A categories that we want to search in the text.
    3. The definition that help you guide to identify the specific label for that category.
    
    TASK (follow the steps in order):
    1. Analyse the text and determine whether the following six Social Determinants of Health (SDOH) are explicitly stated based on the definitions below.  
    2. For each SDOH, select exactly one label from the bracketed list.  
       • If the note contains none of the listed concepts for that SDOH, choose “unknown”.  
       • Do NOT invent new labels, combine labels, or add explanations.  
    3. Return your answer only as a JSON object with six keys (one per SDOH) and values equal to the chosen label.  
       - Do not include any additional text or commentary.
    
    SDOH term and its definition:
    1. {sdoh_def}
    
    OUTPUT TEMPLATE (use exactly this format):
      "{sdoh}": "…"
    
    ---
    Now analyze the following clinical note:
    
    \"\"\"
    {note_text}
    \"\"\"
    """

def sdh_prompt_for_t5(note_text):
    task_prefix = "Extract social determinants of health variables:"
    prompt_content = f"""
    Analyze the following clinical note and indicate whether each of the following seven social determinants of health (SDH) is specifically mentioned:
    
    1. **Employment status**: Whether the patient is currently employed, unemployed, retired, on disability, or has a job title or income source.
    2. **Housing issues**: Any mention of homelessness, unstable housing, living in shelters, or housing concerns (e.g., can't afford rent, frequent moves).
    3. **Transportation needs**: Any reference to transportation difficulties, lack of car access, reliance on public transit, missed appointments due to transportation.
    4. **Parental status**: Whether the patient has children or dependents, or is a caregiver to minors.
    5. **Relationship status**: Whether the patient is married, divorced, single, has a partner, or is widowed.
    6. **Social support**: Social support refers to any documented involvement of a social worker or case manager who provides formal services to the patient. This may include advocacy for patient rights, assessment and intervention to address social, emotional, or financial needs, counseling or therapy, case management such as coordinating services and connecting the patient to housing, insurance, or transportation resources, and facilitating access to community-based programs. Social support is identified when the patient is explicitly assisted by professionals in these roles, often noted as “social work consulted,” “case management involved,” or similar phrases. It does not include informal or emotional support from family members, friends, or romantic partners unless such support is clearly mediated through a formal care plan by a social worker or case manager.
    7. **Substance Use**: Any mention of alcohol, drug, or tobacco use, including current use, past use, or explicit denial of use.
    
    Answer with "Yes" or "No" for each item, and include a short evidence sentence. If not mentioned, say: *There is no evidence.*
    
    Clinical Note:
    {note_text}
    
    Output Format:
    SDH_Employment status_: [Yes/No] - [short evidence sentence]
    SDH_Housing issues: [Yes/No] - [short evidence sentence]
    SDH_Transportation needs: [Yes/No] - [short evidence sentence]
    SDH_Parental status: [Yes/No] - [short evidence sentence]
    SDH_Relationship status: [Yes/No] - [short evidence sentence]
    SDH_Social support: [Yes/No] - [short evidence sentence]
    SDH_Substance Use: [Yes/No] - [short evidence sentence]
    """
    return f"{task_prefix} {prompt_content}"

def remove_square_brackets(text):
    """Elimina todo lo que está entre corchetes cuadrados, dejando solo texto natural."""
    if not isinstance(text, str):
        return None
    
    # Eliminar contenido entre [ y ] (incluyéndolos)
    cleaned = re.sub(r'\[.*?\]', '', text, flags=re.DOTALL)
    
    # Limpiar espacios múltiples y saltos de línea
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None

def save_to_jsonl(data_list, model_id, timestamp):
    filename=f"/data/salazarda/data/sdoh/outputs/sdoh_outputs_{model_id.replace('/data/salazarda/data/models/', '')}_{timestamp}.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry) + "\n")

    df = pd.read_json(filename, lines=True)
    natural_texts = df['text'].apply(remove_square_brackets).dropna().drop_duplicates()
    natural_texts.to_csv(filename.replace('.jsonl', '_natural.csv'), index=False)



def collapse_onehot_group(df, prefix):
    grp_cols = [c for c in df.columns if c.startswith(prefix + "_")]
    def pick_label(row):
        for c in grp_cols:
            if row[c] == 1:
                return c.split("_", 1)[1]
        return 0
    return df.apply(pick_label, axis=1)





### OLD CODE
    # 1. Employment status: whether the patient is currently employed, unemployed (e.g., out of work, jobless, laid-off, etc.), underemployed (e.g., low-paying job, part-time worker, etc.), has a disability (currently not working due to a disability, e.g., unable to work because of a physical or mental health condition, receiving disability benefits, or described as disabled or incapacitated), retired (e.g., inactive, retired person, pensioner, etc.), or a student (e.g., attends college/university, graduate/undergraduate student, etc.). Unknown if no information regarding the patient's current employment status is mentioned at the time of the clinical note. LABELS: [employed, unemployed, underemployed, disability, retired, student, unknown]
    # 2. Housing issues: whether the patient's housing situation is affected by their financial status (e.g., unable to pay rent, facing eviction due to financial hardship, etc.), undomiciled (currently without a fixed dwelling, e.g., homeless, living in a shelter, etc.), or other (the housing situation does not clearly fall under financial status or undomiciled but still provides information about housing, e.g., living in transitional housing, incarceration, or other arrangements not otherwise specified). Unknown if no information regarding the patient's housing situation is mentioned at the time of the clinical note. LABELS: [financial_status, undomiciled, other, unknown]
    # 3. Transportation issues: whether the patient has transportation difficulties related to distance (e.g., the clinic is far from home, requires extended travel time, or is located in a different city or rural area with limited access), resources (e.g., lack of a personal vehicle, inability to afford public transit, missed appointments due to transportation issues, or reliance on others for transportation), or other (factors not covered by distance or resources). Unknown if no information regarding the patient's transportation situation is mentioned at the time of the clinical note. LABELS: [distance, resources, other, unknown]
    # 4. Parental status: whether the patient has a child under 18 years old (yes), does not have a child under 18 years old (no), or if no information regarding the patient's parental status is mentioned at the time of the clinical note (unknown). LABELS: [yes, no, unknown]
    # 5. Relationship status: whether the patient is single, married, divorced, widowed, or partnered, based on information mentioned in the clinical note. Unknown if no information regarding the patient's relationship status is provided at the time of the clinical note. LABELS: [single, married, divorced, widowed, partnered, unknown]
    # 6. Social support: whether the patient has informal or emotional support from family members, friends, or romantic partners (presence), does not have such support (absence), or if no information regarding the patient's informal or emotional support is mentioned at the time of the clinical note (unknown). LABELS: [presence, absence, unknown]
