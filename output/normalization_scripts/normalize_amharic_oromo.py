import pandas as pd
import os
import unicodedata
import re

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV   = os.path.join("output", "amharic_oromo_translations.csv")
OUTPUT_DIR  = "output"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "normalized_amharic_oromo.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column names ──────────────────────────────────────────────────────────────
COL_AMHARIC = "Amharic"
COL_OROMO   = "Afan Oromo"

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
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\.{3,}", "...", text)
    return text

def clean_amharic(text):
    if not isinstance(text, str):
        return text
    text = normalize_homophones(text)
    text = standardize_punctuation(text)
    text = text.replace(".", "።")
    return text

def remove_noise(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"[^\w\sሀ-፿ᎀ-᎟።?!\"',.-]", "", text)
    return text

# ── Load ──────────────────────────────────────────────────────────────────────
def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}") from e
    df.rename(columns={"Oromo": "Afan Oromo"}, inplace=True)
    print(f"Loaded {len(df)} rows from {path}")
    return df

df_raw = load_dataset(INPUT_CSV)
print("\nBefore normalisation (first 5 rows):")
print(df_raw.head(5).to_string(index=False))

# ── Step 1 — Remove duplicates ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 1: Removing duplicates...")
print("=" * 70)
print(f"Rows before deduplication: {len(df_raw)}")
df = df_raw.drop_duplicates().reset_index(drop=True)
print(f"Rows after deduplication: {len(df)}")

# ── Step 2 — Homophonic normalisation (Amharic only) ────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Applying homophonic normalisation (Amharic)...")
print("=" * 70)
df[COL_AMHARIC] = df[COL_AMHARIC].apply(normalize_homophones)
print("✓ Homophonic normalisation complete")

# ── Step 3 — Standardise punctuation ──────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3: Standardising punctuation...")
print("=" * 70)
df[COL_AMHARIC] = df[COL_AMHARIC].apply(standardize_punctuation)
df[COL_OROMO] = df[COL_OROMO].apply(standardize_punctuation)
print("✓ Punctuation standardisation complete")

# ── Step 4 — Case standardization ────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: Case standardization...")
print("=" * 70)
df[COL_AMHARIC] = df[COL_AMHARIC].str.strip()
df[COL_OROMO] = df[COL_OROMO].str.strip()
print("✓ Case standardization complete")

# ── Step 5 — Noise removal ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: Removing noise...")
print("=" * 70)
df[COL_AMHARIC] = df[COL_AMHARIC].apply(remove_noise)
df[COL_OROMO] = df[COL_OROMO].apply(remove_noise)
df = df.dropna().reset_index(drop=True)
print("✓ Noise removal complete")

# ── Final cleanup ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CLEANED DATASET")
print("=" * 70)
print(f"Total sentence pairs: {len(df)}")
print("\nFirst 10 rows:")
print("-" * 70)
print(df.head(10).to_string(index=True))
print("=" * 70)

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved → {OUTPUT_CSV}")
