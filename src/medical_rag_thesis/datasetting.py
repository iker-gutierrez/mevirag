# -*- coding: utf-8 -*-
"""datasetting.py

This script implements a regex-based pipeline to construct a structured clinical QA dataset from TXT guidebooks. The process includes text cleaning, segmentation using regular expressions, filtering, and evaluation.

## 0. Imports and data

Import the required libraries:
"""

import csv
import re
from pathlib import Path

import pandas as pd

"""Define the list of clinical guidebook files to be processed.

The TXT files themselves are not stored in this repository, they are
published in the SNS1064-dataset repository
(https://github.com/iker-gutierrez/SNS1064-dataset) as
`clinical_guidebooks_txt.zip`. Fetch and unzip them into
`data/raw/clinical_guidebooks_txt/` before running this script:

    mkdir -p data/raw/clinical_guidebooks_txt
    curl -L -o /tmp/clinical_guidebooks_txt.zip \\
        https://github.com/iker-gutierrez/SNS1064-dataset/raw/main/clinical_guidebooks_txt.zip
    unzip -oj /tmp/clinical_guidebooks_txt.zip "clinical_guidebooks_txt/*.txt" \\
        -d data/raw/clinical_guidebooks_txt

(-j flattens the zip's own nested clinical_guidebooks_txt/ subdirectory and
the "*.txt" filter skips its __MACOSX/ metadata entries.)
"""

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
GUIDEBOOKS_DIR = DATA_DIR / "clinical_guidebooks_txt"

files = [
    GUIDEBOOKS_DIR / "ansiedad.txt",
    GUIDEBOOKS_DIR / "atencion_paliativa.txt",
    GUIDEBOOKS_DIR / "cuidados_paliativos_pediatria.txt",
    GUIDEBOOKS_DIR / "diabetes.txt",
    GUIDEBOOKS_DIR / "manejo_ictus.txt",
    GUIDEBOOKS_DIR / "prevencion_secundaria_ictus.txt",
]

"""## 1. Text cleaning

Each line is normalized to reduce noise introduced during PDF-to-TXT conversion. This includes trimming whitespace and removing leading bullet-like characters.
"""

def clean_text(t):
    t = t.strip() #To strip leading and trailing whitespaces.
    t = re.sub(r'^[\•\●\-\*]+\s*', '', t)
    t = re.sub(r'^\$?\\bullet\$?\s*', '', t)
    return t.strip() #To apply final whitespace trimming after normalization.

"""## 2. Text segmentation

Regular expressions are used to detect structural elements in the text, including:

- topic
- question
- subquestion
- judgement
- evidence
- considerations

These patterns drive the rule-based extraction process.
"""

pregunta_pattern = re.compile(r'^[a-z]\)')
subpregunta_pattern = re.compile(r'^[a-z]\.\d+\.')
contexto_pattern = re.compile(r'^Contexto', re.IGNORECASE)

juicio_pattern = re.compile(r'Juicio:\s*', re.IGNORECASE)
evidencia_pattern = re.compile(r'Evidencia procedente de la investigación:\s*', re.IGNORECASE)
consideraciones_pattern = re.compile(
    r'^(Consideraciones adicionales|Información adicional):\s*',
    re.IGNORECASE
)

tema_start_pattern = re.compile(
    r'^(Pregunta|Pregunta para responder|Pregunta a responder|Subpregunta)',
    re.IGNORECASE
)

"""## 3. Dataframe creation

The text is processed line by line and segmented into structured QA instances. Segmentation logic relies on local patterns such as:

- "Juicio:"
- "Evidencia procedente de la investigación:"
- "a)", "a.1."

These markers typically appear at the beginning of lines in the TXT files Hence, per-line processing makes it easier to:

- detect section boundaries  
- switch modes (e.g., judgement, evidence)  
- accumulate text until the next trigger appears


Each instance is stored as a row with fields for topic, question, answers, and metadata.
"""

rows = []

for filepath in files:

    filename = filepath.name

    with open(filepath, encoding="utf8") as f:
        texts = [clean_text(line) for line in f if line.strip()]

    current_tema = ""
    current_pregunta = ""
    current_subpregunta = ""

    juicio = []
    evidencia = []
    consideraciones = []

    mode = None
    collecting_tema = False
    tema_buffer = []

    for text in texts:

        # -------- topic start --------
        if tema_start_pattern.match(text):
            collecting_tema = True
            tema_buffer = []
            continue

        # -------- topic end --------
        if collecting_tema and contexto_pattern.match(text):
            collecting_tema = False
            current_tema = " ".join(tema_buffer).strip()
            continue

        if collecting_tema:
            tema_buffer.append(text)
            continue

        # -------- pregunta --------
        if pregunta_pattern.match(text):

            if juicio or evidencia or consideraciones:
                rows.append({
                    "guidebook": filename,
                    "topic": current_tema,
                    "question": current_pregunta,
                    "subquestion": current_subpregunta,
                    "judgement": " ".join(juicio),
                    "evidence": " ".join(evidencia),
                    "considerations": " ".join(consideraciones)
                })

            current_pregunta = text
            current_subpregunta = ""

            juicio = []
            evidencia = []
            consideraciones = []
            mode = "question"

            continue

        # -------- subpregunta --------
        if subpregunta_pattern.match(text):

            if juicio or evidencia or consideraciones:
                rows.append({
                    "guidebook": filename,
                    "topic": current_tema,
                    "question": current_pregunta,
                    "subquestion": current_subpregunta,
                    "judgement": " ".join(juicio),
                    "evidence": " ".join(evidencia),
                    "considerations": " ".join(consideraciones)
                })

            current_subpregunta = text

            juicio = []
            evidencia = []
            consideraciones = []
            mode = None

            continue

        # -------- juicio --------
        if juicio_pattern.search(text):
            mode = "judgement"
            text = juicio_pattern.split(text, 1)[1]
            juicio.append(text)
            continue

        # -------- evidencia --------
        if evidencia_pattern.search(text):
            mode = "evidence"
            text = evidencia_pattern.split(text, 1)[1]
            evidencia.append(text)
            continue

        # -------- consideraciones --------
        if consideraciones_pattern.search(text):
            mode = "considerations"
            text = consideraciones_pattern.split(text, 1)[1]
            consideraciones.append(text)
            continue

        # -------- accumulate --------
        if mode == "question":
            current_pregunta = (current_pregunta + " " + text).strip()

        elif mode == "judgement":
            juicio.append(text)

        elif mode == "evidence":
            evidencia.append(text)

        elif mode == "considerations":
            consideraciones.append(text)

    # -------- save last row --------
    if juicio or evidencia or consideraciones:
        rows.append({
            "guidebook": filename,
            "topic": current_tema,
            "question": current_pregunta,
            "subquestion": current_subpregunta,
            "judgement": " ".join(juicio),
            "evidence": " ".join(evidencia),
            "considerations": " ".join(consideraciones)
        })

df = pd.DataFrame(rows)

"""## 4. Sample filtering

Noisy samples are removed using regex. In particular, entries with placeholder evidence are excluded: *"ver apartado(s) anterior(es)"*, which means, "see previous section(s)".
"""

# Regex for skipping samples containing "ver apartado(s) anterior(es)":
skip_pattern = re.compile(r"ver apartado(s)? anterior(es)?", re.IGNORECASE)

filtered_df = df[~df["evidence"].str.contains(skip_pattern, na=False)]

print("Original:", len(df))
print("Filtered:", len(filtered_df))

try:
    filtered_df.to_csv(DATA_DIR / "dataset_noncurated.csv", index=False, encoding="utf8")
    print("CSV created successfully")
except Exception as e:
    print("Error while saving CSV:", e)

"""## 5. Intermediate automatic evaluation

Quality checks are performed on the extracted dataset, including:

- Presence rate of optional features
- Presence rate of mandatory features
"""

#Checkpoint
import re
import pandas as pd

filtered_df = pd.read_csv(DATA_DIR / "dataset_noncurated.csv")

"""### 5.1 Presence rate of optional features

Let's measure how often optional features (subquestion, considerations) appear in the dataset:
"""

subq_ratio = filtered_df["subquestion"].notna().mean()
cons_ratio = filtered_df["considerations"].notna().mean()

print("subquestion presence (%):", subq_ratio*100)
print("considerations presence (%):", cons_ratio*100)

"""### 5.2 Presence rate of mandatory features

Let's verify if all cells of required features (e.g., question, judgement, evidence) are non-empty across all samples:
"""

required_cols = ["guidebook", "topic", "question", "judgement", "evidence"]

for col in required_cols:
    valid = filtered_df[col].notna() & (filtered_df[col].astype(str).str.strip() != "")
    percentage = valid.mean() * 100
    print(f"{col} presence (%): {percentage:.2f}")

"""The output cell above shows that there are empty cells in the `judgement` and `evidence` features."""

# True if cell is valid (non-empty, non-null)
valid_mask = (
    filtered_df[required_cols]
    .notna()
    & (filtered_df[required_cols].astype(str).apply(lambda x: x.str.strip() != ""))
).all(axis=1)

# indices of incomplete samples
incomplete_indices = filtered_df.index[~valid_mask].tolist()

print("Number of incomplete samples:", len(incomplete_indices))
print("Indices:")
print(incomplete_indices)

"""The output cell above shows that there are 11 empty cells in mandatory features. The indices of those samples are provided for later manual curation (§7).

## 6. Train/dev/test split

The dataset is split into training, development, and test sets, stratified
by guidebook. Dev and test are sized to match CasiMédicos-Exp's own dev
(63) and test (125) splits exactly (`docs/curated_splits_reproduction.md`,
step 3), not an 80/20 train/test split and not a fraction-based split
rounded to the nearest row -- `train_test_split` is called twice with exact
integer sizes, so the absolute dev/test counts land at 63/125 regardless of
how many rows the corrected extraction (§1-4) produces.

Requires `dataset.csv`, which must exist by this point: `dataset_noncurated.csv`
was written by §4, evaluated in §5, and is expected to have been renamed to
`dataset.csv` in between -- this notebook does not do that renaming itself.
"""

from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA_DIR / "dataset.csv")

DEV_SIZE = 63
TEST_SIZE = 125

train_dev_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=42,
    stratify=df["guidebook"],
)
train_df, dev_df = train_test_split(
    train_dev_df,
    test_size=DEV_SIZE,
    random_state=42,
    stratify=train_dev_df["guidebook"],
)

print(f"Train size: {len(train_df)}, Dev size: {len(dev_df)}, Test size: {len(test_df)}")
assert len(dev_df) == DEV_SIZE and len(test_df) == TEST_SIZE

train_df = train_df.sort_values(by="guidebook").reset_index(drop=True) # sort by guidebook
train_df.to_csv(DATA_DIR / "train.csv", index=False)

dev_df = dev_df.sort_values(by="guidebook").reset_index(drop=True) # sort by guidebook
dev_df.to_csv(DATA_DIR / "dev.csv", index=False)

test_df = test_df.sort_values(by="guidebook").reset_index(drop=True) # sort by guidebook
test_df.to_csv(DATA_DIR / "test.csv", index=False)

df = pd.concat([train_df, dev_df, test_df], ignore_index=True)

# sort by guidebook
df = df.sort_values(by="guidebook").reset_index(drop=True)
df.to_csv(DATA_DIR / "dataset.csv", index=False)

"""## 7. Manual curation

The entire test set is manually reviewed to fix segmentation errors and ensure data quality. However, the training set is only partially inspected, as noise is less critical in training data and full manual curation would be highly time-consuming.

## 8. Final automatic evaluation

After curation, the dataset is re-evaluated to check that the manual curation worked and now all mandatory features are non-empty.
"""

#Checkpoint
import re
import pandas as pd

df = pd.read_csv(DATA_DIR / "dataset.csv")

print(f"Dataset size: {len(df)}")

"""### 8.1 Presence rate of optional features

Let's recompute the distribution of optional fields after manual corrections.
"""

subq_ratio = df["subquestion"].notna().mean()
cons_ratio = df["considerations"].notna().mean()

print("subquestion presence (%):", subq_ratio*100)
print("considerations presence (%):", cons_ratio*100)

"""### 8.2. Presence rate of mandatory features

Final evaluation confirms 100% presence for all mandatory features.
"""

required_cols = ["guidebook", "topic", "question", "judgement", "evidence"]

for col in required_cols:
    valid = df[col].notna() & (df[col].astype(str).str.strip() != "")
    percentage = valid.mean() * 100
    print(f"{col} presence (%): {percentage:.2f}")

# True if cell is valid (non-empty, non-null)
valid_mask = (
    df[required_cols]
    .notna()
    & (df[required_cols].astype(str).apply(lambda x: x.str.strip() != ""))
).all(axis=1)

# indices of incomplete samples
incomplete_indices = df.index[~valid_mask].tolist()

print("Number of incomplete samples:", len(incomplete_indices))
print("Indices:")
print(incomplete_indices)

"""## 9. Quantitative analysis

Now that the dataset is ready, let's measure its quantative characteristics, including number of sentences, tokens, and per-feature sentence lengths.

### 9.1. Number of sentences and word tokens

Total sentence and word token counts are computed across all textual features:
"""

import re
import pandas as pd

def count_sentences(text):
    if pd.isna(text):
        return 0
    sentences = re.split(r'[.!?]+', str(text))
    return len([s for s in sentences if s.strip()])

def count_tokens(text):
    if pd.isna(text):
        return 0
    # extract word tokens (letters + digits, no punctuation)
    tokens = re.findall(r'\b\w+\b', str(text))
    return len(tokens)

# apply over relevant columns
text_cols = ["question", "subquestion", "judgement", "evidence", "considerations"]

df["sentences"] = df[text_cols].fillna("").apply(
    lambda row: sum(count_sentences(x) for x in row), axis=1
)

df["tokens"] = df[text_cols].fillna("").apply(
    lambda row: sum(count_tokens(x) for x in row), axis=1
)

total_sentences = df["sentences"].sum()
total_tokens = df["tokens"].sum()

print("Sentences:", total_sentences)
print("Tokens:", total_tokens)

"""### 9.2. Sentence length

Average and standard deviation of token lengths is measured for each feature:
"""

# --- lengths ---
df["q_len"] = df["question"].str.split().str.len()
df["e_len"] = df["evidence"].str.split().str.len()
df["j_len"] = df["judgement"].str.split().str.len()
df["s_len"] = df["subquestion"].str.split().str.len()
df["c_len"] = df["considerations"].str.split().str.len()

# --- question ---
print("Avg question length (tokens):", df["q_len"].mean())
print("Std question length (tokens):", df["q_len"].std())

# --- evidence ---
print("Avg evidence length (tokens):", df["e_len"].mean())
print("Std evidence length (tokens):", df["e_len"].std())

# --- judgement ---
print("Avg judgement length (tokens):", df["j_len"].mean())
print("Std judgement length (tokens):", df["j_len"].std())

# --- subquestion (non-empty only) ---
subq_mask = df["subquestion"].notna() & (df["subquestion"].str.strip() != "")
print("Avg subquestion length (tokens):", df.loc[subq_mask, "s_len"].mean())
print("Std subquestion length (tokens):", df.loc[subq_mask, "s_len"].std())

# --- considerations (non-empty only) ---
cons_mask = df["considerations"].notna() & (df["considerations"].str.strip() != "")
print("Avg considerations length (tokens):", df.loc[cons_mask, "c_len"].mean())
print("Std considerations length (tokens):", df.loc[cons_mask, "c_len"].std())

"""### 9.3. Samples per guidebook

Distribution of samples across the different clinical guidebooks:
"""

print("Samples per guidebook:")
print(df["guidebook"].value_counts())