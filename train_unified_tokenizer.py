"""
Train a unified multilingual SentencePiece tokenizer across all 3 languages.

RATIONALE:
  - Single tokenizer eliminates vocabulary redundancy between per-language models
  - Better cross-lingual alignment (shared subwords for cognates)
  - Simpler unified vocabulary (no duplicates)
  - Potentially better translation quality through improved morphological sharing

OUTPUT:
  - output/spm_unified_multilingual.model
  - output/spm_unified_multilingual.vocab
"""

import os
import tempfile
import sentencepiece as spm
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"
SAVE_PREFIX = os.path.join(OUTPUT_DIR, "spm_unified_multilingual")

# Data sources (all available parallel datasets)
DATA_SOURCES = [
    ("output/clean_translation.csv", "amharic_english_oromo"),      # ~2.5k triples
    ("output/normalized_age_amharic_english.csv", "amharic_english"), # 17.3k pairs
    ("output/normalized_amharic_oromo.csv", "amharic_oromo"),         # 144.3k pairs
]

# Unified tokenizer vocabulary size
# Calculated as: (all_tokens_across_languages) × 0.8-1.0
# Conservative estimate: 15,000 tokens (handles all morphology + cross-lingual coverage)
VOCAB_SIZE = 15000

# SentencePiece hyperparameters
CHARACTER_COVERAGE = 0.9995  # Keep all Ethiopic glyphs and Latin
MODEL_TYPE = "bpe"
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

print("=" * 70)
print("UNIFIED MULTILINGUAL SENTENCEPIECE TOKENIZER")
print("=" * 70)

# ── Step 1: Collect all text from all datasets ────────────────────────────────
print("\nSTEP 1: Loading all available text from parallel datasets...")
print("-" * 70)

all_text = []
total_pairs = 0

for csv_path, description in DATA_SOURCES:
    if not os.path.isfile(csv_path):
        print(f"⚠ Skipping {csv_path} (not found)")
        continue

    # Determine separator (tab or comma)
    if "clean_translation.csv" in csv_path:
        sep = "\t"
    else:
        sep = ","

    try:
        df = pd.read_csv(csv_path, sep=sep)
        # Collect all text columns (skip metadata)
        for col_idx, col in enumerate(df.columns):
            text_series = df.iloc[:, col_idx].dropna()
            all_text.extend(text_series.astype(str).tolist())
            total_pairs += len(df)
        print(f"✓ {description:30s} ({csv_path})")
        print(f"  Rows: {len(df):,} | Total text lines collected: {len(all_text):,}")
    except Exception as e:
        print(f"✗ Failed to load {csv_path}: {e}")

print(f"\nTotal text samples: {len(all_text):,}")
print(f"Total parallel pairs processed: {total_pairs:,}")

# ── Step 2: Write combined text to temporary file ───────────────────────────
print("\n" + "-" * 70)
print("STEP 2: Creating temporary training corpus...")

with tempfile.NamedTemporaryFile(
    mode="w", suffix=".txt", delete=False, encoding="utf-8"
) as f:
    f.write("\n".join(all_text))
    tmp_path = f.name

print(f"✓ Temporary corpus created: {tmp_path}")
print(f"  Total lines: {len(all_text):,}")

# ── Step 3: Train unified SentencePiece model ──────────────────────────────
print("\n" + "-" * 70)
print("STEP 3: Training unified multilingual SentencePiece model...")
print(f"  Vocabulary size: {VOCAB_SIZE:,}")
print(f"  Character coverage: {CHARACTER_COVERAGE}")
print(f"  Model type: {MODEL_TYPE}")
print("  (This may take a few minutes...)")

os.makedirs(OUTPUT_DIR, exist_ok=True)

spm.SentencePieceTrainer.train(
    input=tmp_path,
    model_prefix=SAVE_PREFIX,
    vocab_size=VOCAB_SIZE,
    model_type=MODEL_TYPE,
    character_coverage=CHARACTER_COVERAGE,
    pad_id=PAD_ID,
    unk_id=UNK_ID,
    bos_id=BOS_ID,
    eos_id=EOS_ID,
    # Additional parameters for quality
    hard_vocab_limit=False,
    normalization_rule_name="nmt_nfkc",
)

print("✓ Training complete!")

# ── Step 4: Load and verify ────────────────────────────────────────────────
print("\n" + "-" * 70)
print("STEP 4: Verifying unified tokenizer...")

sp = spm.SentencePieceProcessor()
sp.load(f"{SAVE_PREFIX}.model")

vocab_size = sp.get_piece_size()
print(f"✓ Unified vocabulary size: {vocab_size:,}")

# Test encoding with all three languages
test_samples = {
    "Amharic": "ሰላም ዓለም",
    "Oromo": "Salaam addunyaa",
    "English": "Hello world"
}

print("\nTest tokenization:")
for lang, text in test_samples.items():
    tokens = sp.encode(text, out_type=str)
    print(f"  {lang:10s}: {text:30s} → {tokens}")

# ── Step 5: Cleanup ────────────────────────────────────────────────────────
print("\n" + "-" * 70)
print("STEP 5: Cleanup...")
os.unlink(tmp_path)
print("✓ Temporary files cleaned up")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("UNIFIED TOKENIZER READY")
print("=" * 70)
print(f"Model: {SAVE_PREFIX}.model")
print(f"Vocab: {SAVE_PREFIX}.vocab")
print(f"Vocabulary size: {vocab_size:,}")
print("=" * 70)
