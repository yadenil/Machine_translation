import pandas as pd
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT_CSV   = os.path.join("output", "AGE_amharic_english.csv")
OUTPUT_DIR  = "output"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "normalized_age_amharic_english.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Column names ──────────────────────────────────────────────────────────────
COL_AMHARIC = "Amharic"
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

def remove_outer_quotes(text):
    """Remove outer quotes (both single and double) from text."""
    if not isinstance(text, str):
        return text
    text = text.strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text

def normalize_amharic(text):
    if not isinstance(text, str):
        return text
    text = normalize_homophones(text)
    return text

def normalize_english(text):
    if not isinstance(text, str):
        return text
    text = remove_outer_quotes(text)
    text = text.lower()
    return text

# ── Load ──────────────────────────────────────────────────────────────────────
def load_dataset(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}") from e
    print(f"Loaded {len(df)} rows from {path}")
    return df

df = load_dataset(INPUT_CSV)
print("\nBefore normalization (first 5 rows):")
print(df.head(5).to_string(index=False))

# ── Normalization ────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NORMALIZATION")
print("=" * 70)
print("✓ Applying homophone normalization to Amharic...")
print("✓ Removing outer quotes from English...")
print("✓ Converting English to lowercase...")

df[COL_AMHARIC] = df[COL_AMHARIC].apply(normalize_amharic)
df[COL_ENGLISH] = df[COL_ENGLISH].apply(normalize_english)

print(f"✓ Normalization complete")

# ── Print normalized dataset ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("NORMALIZED DATASET")
print("=" * 70)
print(f"Total sentence pairs: {len(df)}")
print("\nFirst 10 rows:")
print("-" * 70)
print(df.head(10).to_string(index=True))
print("=" * 70)

# ── Save as CSV ───────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved → {OUTPUT_CSV}")
print(f"✓ Format: CSV (comma-separated columns)")
