import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import sentencepiece as spm
import tempfile, os

# ── Load clean data ────────────────────────────────────────────────────────────
df = pd.read_csv("output/clean_translation.csv", sep="\t")
df.columns = ["Amharic", "Afan_Oromo", "English"]
df = df.dropna().reset_index(drop=True)

# ── Punctuation removal ────────────────────────────────────────────────────────
ETHIOPIC_PUNCT = "።፣፤፥፦፧፨፡"
ASCII_PUNCT    = string.punctuation

def clean_general(text):
    punct = ASCII_PUNCT + ETHIOPIC_PUNCT
    return text.translate(str.maketrans("", "", punct))

def clean_oromo(text):
    """Remove all punctuation except apostrophe (hauda / glottal-stop marker)."""
    punct = ASCII_PUNCT.replace("'", "") + ETHIOPIC_PUNCT
    return text.translate(str.maketrans("", "", punct))

df["Amharic_c"]    = df["Amharic"].apply(clean_general)
df["Afan_Oromo_c"] = df["Afan_Oromo"].apply(clean_oromo)
df["English_c"]    = df["English"].apply(clean_general)

# ── Subword (BPE) tokenization via SentencePiece ──────────────────────────────
# Amharic uses a large Ethiopic syllabary → higher vocab budget
# Afan Oromo and English are Latin-based → smaller vocab sufficient
VOCAB_AM = 2000
VOCAB_OR = 1500
VOCAB_EN = 1500

def train_spm(series, model_prefix, vocab_size):
    """Train a SentencePiece BPE model on the given text series."""
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        f.write("\n".join(series.astype(str).tolist()))
        tmp_path = f.name
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,   # high coverage keeps all Ethiopic glyphs
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )
    os.unlink(tmp_path)
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

print("Training BPE models …")
sp_am = train_spm(df["Amharic_c"],    "output/spm_amharic", VOCAB_AM)
sp_or = train_spm(df["Afan_Oromo_c"], "output/spm_oromo",   VOCAB_OR)
sp_en = train_spm(df["English_c"],    "output/spm_english",  VOCAB_EN)
print("BPE models ready.")

def subword_token_count(series, sp):
    """Number of subword tokens per sentence."""
    return series.apply(lambda s: len(sp.encode(str(s), out_type=str)))

df["stc_am"] = subword_token_count(df["Amharic_c"],    sp_am)
df["stc_or"] = subword_token_count(df["Afan_Oromo_c"], sp_or)
df["stc_en"] = subword_token_count(df["English_c"],    sp_en)

# Word counts (whitespace split)
df["wc_am"] = df["Amharic_c"].apply(lambda s:    len(str(s).split()))
df["wc_or"] = df["Afan_Oromo_c"].apply(lambda s: len(str(s).split()))
df["wc_en"] = df["English_c"].apply(lambda s:    len(str(s).split()))

# Token-to-Word Ratio per sentence; mean across corpus
twr = {
    "Amharic":    (df["stc_am"] / df["wc_am"].replace(0, np.nan)).mean(),
    "Afan Oromo": (df["stc_or"] / df["wc_or"].replace(0, np.nan)).mean(),
    "English":    (df["stc_en"] / df["wc_en"].replace(0, np.nan)).mean(),
}

# ── Styling ───────────────────────────────────────────────────────────────────
COLORS = ["#e05c5c", "#5b9bd5", "#4caf6e"]
LANGS  = list(twr.keys())
RATIOS = list(twr.values())

# ── Bar chart ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

bars = ax.bar(LANGS, RATIOS, color=COLORS, width=0.5, zorder=3)

# Value labels on top of each bar
for bar, ratio in zip(bars, RATIOS):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{ratio:.2f}",
        ha="center", va="bottom",
        color="white", fontsize=13, fontweight="bold"
    )

ax.set_ylabel("Avg Subword Tokens per Word", color="#aaaaaa", fontsize=11)
ax.set_xlabel("Language", color="#aaaaaa", fontsize=11)
ax.tick_params(colors="#aaaaaa", labelsize=11)
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")
ax.grid(axis="y", color="#2a2d3a", linewidth=0.7, zorder=0)
ax.set_ylim(0, max(RATIOS) * 1.25)

ax.set_title(
    "Token-to-Word Ratio — BPE Subword Tokens per Word\n"
    f"(AM vocab={VOCAB_AM:,}  |  OR vocab={VOCAB_OR:,}  |  EN vocab={VOCAB_EN:,}  "
    f"|  {len(df):,} sentence pairs)",
    color="white", fontsize=12, fontweight="bold", pad=12
)

# Summary stat line
summary = (
    f"Avg words/sentence — AM: {df['wc_am'].mean():.1f}  "
    f"OR: {df['wc_or'].mean():.1f}  EN: {df['wc_en'].mean():.1f}   |   "
    f"Avg subword tokens/sentence — AM: {df['stc_am'].mean():.1f}  "
    f"OR: {df['stc_or'].mean():.1f}  EN: {df['stc_en'].mean():.1f}"
)
fig.text(0.5, 0.01, summary, ha="center", va="bottom", color="#cccccc",
         fontsize=8, fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1d27", edgecolor="#333344"))

out = "translation_analysis.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}")
plt.show()
