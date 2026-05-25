import pandas as pd
import os
import unicodedata
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
EXCEL_PATH  = "text.xlsx"
OUTPUT_DIR  = "output"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "clean_translation.csv")
OUTPUT_XLSX = os.path.join(OUTPUT_DIR, "normalized_translations.xlsx")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column names ──────────────────────────────────────────────────────────────
COL_AMHARIC = "Amharic"
COL_OROMO   = "Afan Oromo"
COL_ENGLISH = "English"

# ── Homophone normalisation map (Amharic only) ────────────────────────────────
homophone_map = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ኸ": "ሀ", "ኹ": "ሁ", "ኺ": "ሂ", "ኻ": "ሃ", "ኼ": "ሄ", "ኽ": "ህ", "ኾ": "ሆ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ጸ": "ፀ", "ጹ": "ፁ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ጽ": "ፅ", "ጾ": "ፆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ"
}

def normalize_homophones(text):
    if not isinstance(text, str):
        return text
    for old, new in homophone_map.items():
        text = text.replace(old, new)
    return text

def standardize_punctuation(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r"\.{3,}", "...", text)
    return text

def clean_amharic(text):
    if not isinstance(text, str):
        return text
    text = normalize_homophones(text)
    text = standardize_punctuation(text)
    text = text.replace(".", "።")
    return text

def standardize_sentence_punctuation(amharic, english, oromo):
    is_question = english.strip().endswith("?")
    amharic = re.sub(r"[።?.!]+$", "", amharic.strip())
    english = re.sub(r"[።?.!]+$", "", english.strip())
    oromo   = re.sub(r"[።?.!]+$", "", oromo.strip())
    if is_question:
        amharic += "?"; english += "?"; oromo += "?"
    else:
        amharic += "።"; english += "."; oromo += "."
    return amharic, english, oromo

# ── Load ──────────────────────────────────────────────────────────────────────
def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Excel file not found: {path}")
    try:
        df = pd.read_excel(path, engine="openpyxl", header=None)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}") from e
    df.columns = [COL_AMHARIC, COL_ENGLISH, COL_OROMO]
    print(f"Loaded {len(df)} rows from {path}")
    return df

df_raw = load_dataset(EXCEL_PATH)
print("\nBefore normalisation (first 5 rows):")
print(df_raw.head(5).to_string(index=False))

# ── Step 1 — Normalise ────────────────────────────────────────────────────────
df = df_raw.copy()
df[COL_AMHARIC] = df[COL_AMHARIC].apply(clean_amharic)
df[COL_ENGLISH] = df[COL_ENGLISH].apply(standardize_punctuation)
df[COL_OROMO]   = df[COL_OROMO].apply(standardize_punctuation)

df[[COL_AMHARIC, COL_ENGLISH, COL_OROMO]] = df.apply(
    lambda row: standardize_sentence_punctuation(
        row[COL_AMHARIC], row[COL_ENGLISH], row[COL_OROMO]
    ),
    axis=1, result_type="expand"
)
df = df.drop_duplicates().dropna().reset_index(drop=True)

# Reorder columns: Amharic | Afan Oromo | English
df = df[[COL_AMHARIC, COL_ENGLISH, COL_OROMO]]
df.columns = ["Amharic", "english", "afan oromo"]  # reorder and rename for consistency

# ── Step 2 — Strip spurious double-quotes ─────────────────────────────────────
# Rule: if Amharic has no " in its content the sentence is not dialogue,
# so remove any " that ended up in Afan_Oromo or English (CSV artefacts).
# The 9 dialogue rows have " in all three columns and are left untouched.
no_quote_in_am = ~df["Amharic"].str.contains('"', na=False)
df.loc[no_quote_in_am, "afan oromo"] = (
    df.loc[no_quote_in_am, "afan oromo"].str.replace('"', "", regex=False)
)
df.loc[no_quote_in_am, "english"] = (
    df.loc[no_quote_in_am, "english"].str.replace('"', "", regex=False)
)

# ── Step 3 — Print cleaned translation ───────────────────────────────────────
print("\n" + "=" * 70)
print("CLEANED TRANSLATION DATASET")
print("=" * 70)
print(f"Total sentence pairs : {len(df)}")
print(f'Dialogue rows (quotes in all 3 cols) : {df["Amharic"].str.contains(chr(34)).sum()}')
print(f'Remaining spurious quotes in Oromo : {df["afan oromo"].str.contains(chr(34)).sum()}')
print(f'Remaining spurious quotes in English     : {df["English"].str.contains(chr(34)).sum()}')
print("\nFirst 10 rows:")
print("-" * 70)
print(df.head(10).to_string(index=True))
print("=" * 70)

# ── Step 4 — Save ─────────────────────────────────────────────────────────────
# TSV — tab separator means commas inside fields never need CSV quoting
df.to_csv(OUTPUT_CSV, sep="\t", index=False)
df.to_excel(OUTPUT_XLSX, index=False)

print(f"\nSaved → {OUTPUT_CSV}   (tab-separated, no field quoting)")
print(f"Saved → {OUTPUT_XLSX}")