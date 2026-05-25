"""
Multilingual Translation Model
==============================
Combines 3 SentencePiece tokenizers (Amharic, Afan Oromo, English) into a
single seq2seq Transformer that can translate between any language pair.

Usage:
    python train_translator.py          # train the model
    python train_translator.py --test   # load saved model and translate interactively
"""

import os, sys, math, random
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "output/cleaned_dataset/clean_translation.csv"
MODEL_DIR = "output"
SAVE_PATH = os.path.join(MODEL_DIR, "translator.pt")

# Model hyperparameters (small — suited for ~2k sentence pairs)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
MAX_LEN = 80

# Training
EPOCHS = 60
BATCH_SIZE = 32
LR = 3e-4
LABEL_SMOOTHING = 0.1

# Special token IDs (same across all 3 SPM models per your training config)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Language tags — we'll prepend these to the *decoder* input so the model
# knows which language to generate.
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

# ── Load Unified Multilingual SentencePiece Tokenizer ───────────────────────
# UNIFIED APPROACH (Recommended):
#   Single SentencePiece tokenizer trained on combined corpus of all 3 languages.
#
# Benefits over per-language tokenizers:
#   ✓ No vocabulary redundancy (no duplicate tokens across tokenizers)
#   ✓ Better cross-lingual alignment (shared subwords for cognates)
#   ✓ Simpler unified vocabulary
#   ✓ Potentially better translation quality
#   ✓ Reduced model complexity
#
# Fallback: If unified tokenizer not found, load per-language tokenizers

def load_sp(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

# Try to load the unified multilingual tokenizer (recommended)
unified_tokenizer_path = os.path.join(MODEL_DIR, "spm_unified_multilingual.model")

if os.path.isfile(unified_tokenizer_path):
    print("✓ Loading unified multilingual tokenizer...")
    sp_unified = load_sp(unified_tokenizer_path)
    # All languages use the same unified tokenizer
    SP = {
        "am": sp_unified,
        "or": sp_unified,
        "en": sp_unified
    }
else:
    print("⚠ Unified tokenizer not found. Falling back to per-language tokenizers...")
    # Fallback: use per-language tokenizers (less optimal)
    sp_am = load_sp(os.path.join(MODEL_DIR, "spm_amharic.model"))
    sp_or = load_sp(os.path.join(MODEL_DIR, "spm_oromo.model"))
    sp_en = load_sp(os.path.join(MODEL_DIR, "spm_english.model"))
    SP = {
        "am": sp_am,
        "or": sp_or,
        "en": sp_en
    }

# ── Build vocabulary from unified tokenizer ───────────────────────────────────
# SIMPLIFIED VOCABULARY: All languages use the same tokenizer, so vocabulary is
# straightforward — just the tokens from the unified SentencePiece model.
#
# Vocabulary size: ~15,000 tokens (covers all 3 languages)
# This is smaller than the multi-tokenizer approach (~19k) and has no duplicates.

def build_unified_vocab():
    """
    Build vocabulary from the unified multilingual SentencePiece tokenizer.

    Returns:
        token2id: dict mapping token strings to unified integer IDs
        id2token: dict mapping unified integer IDs back to token strings
    """
    token2id = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
    next_id = 4

    # Add language tags
    for tag in LANG_TAGS.values():
        token2id[tag] = next_id
        next_id += 1

    # Add tokens from the unified tokenizer
    # (all languages use the same tokenizer, so we only iterate once)
    sp = SP["am"]  # Could be any language, they all use the same tokenizer
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)

        # Skip special tokens (they're already in token2id)
        if piece in token2id:
            continue

        # Add new token
        token2id[piece] = next_id
        next_id += 1

    # Reverse mapping
    id2token = {v: k for k, v in token2id.items()}

    # Print vocabulary statistics
    print("\n" + "=" * 70)
    print("VOCABULARY STATISTICS (UNIFIED MULTILINGUAL)")
    print("=" * 70)
    print(f"Tokenizer: spm_unified_multilingual")
    print(f"Base vocabulary: {sp.get_piece_size():,} tokens")
    print(f"Plus language tags: {len(LANG_TAGS)}")
    print(f"Plus special tokens: 4 (<pad>, <unk>, <bos>, <eos>)")
    print(f"Total unified vocabulary: {len(token2id):,} tokens")
    print("=" * 70 + "\n")

    return token2id, id2token

TOKEN2ID, ID2TOKEN = build_unified_vocab()
VOCAB_SIZE = len(TOKEN2ID)
print(f"Unified vocabulary size: {VOCAB_SIZE}")

# ── Encode / Decode helpers ──────────────────────────────────────────────────

def encode_sentence(text, lang, max_len=MAX_LEN):
    """
    Tokenize text using the unified multilingual SentencePiece tokenizer.

    Args:
        text: Input text to tokenize
        lang: Language code ("am", "or", "en") — all use the same tokenizer
        max_len: Maximum sequence length (default MAX_LEN)

    Returns:
        List of token IDs [BOS_ID, ...token_ids..., EOS_ID]

    Example:
        encode_sentence("Hello world", "en")  → [2, token_ids..., 3]
        encode_sentence("ሰላም ዓለም", "am")    → [2, token_ids..., 3]
    """
    sp = SP[lang]

    # Encode text using unified tokenizer
    pieces = sp.encode(str(text), out_type=str)

    # Map SentencePiece tokens to unified vocabulary IDs
    ids = [TOKEN2ID.get(p, UNK_ID) for p in pieces]

    # Truncate to leave room for BOS and EOS tokens
    ids = ids[: max_len - 2]

    return [BOS_ID] + ids + [EOS_ID]

def decode_sentence(ids, lang):
    """Convert unified IDs back to text using the language-specific SPM."""
    sp = SP[lang]
    # Map unified IDs → SPM pieces, skip specials
    pieces = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        token = ID2TOKEN.get(i, "<unk>")
        if token in LANG_TAGS.values():
            continue
        pieces.append(token)
    return sp.decode(pieces)

def pad_sequence(seq, max_len):
    return seq + [PAD_ID] * (max_len - len(seq))

# ── Checkpoint Compatibility ──────────────────────────────────────────────────

def check_checkpoint_compatibility(saved_vocab_size):
    """Verify saved checkpoint vocab matches current tokenizer vocab."""
    if saved_vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Checkpoint vocab {saved_vocab_size} ≠ current {VOCAB_SIZE}. "
            "Retrain from scratch with current tokenizer."
        )

# ── Dataset ──────────────────────────────────────────────────────────────────

class TranslationDataset(Dataset):
    """
    Creates all 6 translation directions from the trilingual data:
      am→en, en→am, am→or, or→am, en→or, or→en
    """
    PAIRS = [("am", "en"), ("en", "am"),
             ("am", "or"), ("or", "am"),
             ("en", "or"), ("or", "en")]

    COL = {"am": "Amharic", "or": "Afan_Oromo", "en": "English"}

    def __init__(self, df, max_len=MAX_LEN):
        self.samples = []
        for src_lang, tgt_lang in self.PAIRS:
            src_col = self.COL[src_lang]
            tgt_col = self.COL[tgt_lang]
            tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
            for _, row in df.iterrows():
                src_ids = encode_sentence(row[src_col], src_lang, max_len)
                tgt_ids = encode_sentence(row[tgt_col], tgt_lang, max_len)
                # Insert target-language tag right after BOS
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]  # replace duplicate BOS
                self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return src, tgt

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)
    src_padded = torch.tensor([pad_sequence(s, max_src) for s in src_batch], dtype=torch.long)
    tgt_padded = torch.tensor([pad_sequence(t, max_tgt) for t in tgt_batch], dtype=torch.long)
    return src_padded, tgt_padded

# ── Positional Encoding ─────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# ── Transformer Model ────────────────────────────────────────────────────────

class Translator(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, dropout, max_len):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def make_pad_mask(self, x):
        return (x == PAD_ID)  # (B, S) — True where padded

    def make_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()

    def forward(self, src, tgt):
        # src: (B, S_src)   tgt: (B, S_tgt)
        src_pad_mask = self.make_pad_mask(src)
        tgt_pad_mask = self.make_pad_mask(tgt)
        tgt_causal = self.make_causal_mask(tgt.size(1))

        src_emb = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=tgt_causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )
        return self.output_proj(out)

# ── Training ─────────────────────────────────────────────────────────────────

def train():
    print(f"Device: {DEVICE}")

    # Load data
    df = pd.read_csv(DATA_PATH, sep="\t")
    df.columns = ["Amharic", "Afan_Oromo", "English"]
    df = df.dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} sentence triples")

    # Train/val split (90/10)
    n_val = max(1, len(df) // 10)
    indices = list(range(len(df)))
    random.seed(42)
    random.shuffle(indices)
    val_idx, train_idx = indices[:n_val], indices[n_val:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)

    train_ds = TranslationDataset(df_train)
    val_ds = TranslationDataset(df_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    print(f"Training samples: {len(train_ds)} | Validation samples: {len(val_ds)}")

    # Model
    model = Translator(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, MAX_LEN).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    def calculate_accuracy(logits, targets, pad_id=PAD_ID):
        """Calculate token-level accuracy (ignoring padding tokens)."""
        predictions = logits.argmax(dim=-1)
        mask = targets != pad_id
        correct = (predictions == targets) & mask
        return correct.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0

    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for src, tgt in train_dl:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            mask = tgt_out != PAD_ID
            train_correct += ((predictions == tgt_out) & mask).sum().item()
            train_total += mask.sum().item()

        scheduler.step()
        train_loss /= len(train_dl)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ── Validate ──
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt in val_dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
                val_loss += loss.item()

                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                mask = tgt_out != PAD_ID
                val_correct += ((predictions == tgt_out) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_dl)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "token2id": TOKEN2ID,
                "id2token": ID2TOKEN,
                "config": {
                    "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                    "d_ff": D_FF, "dropout": DROPOUT, "max_len": MAX_LEN,
                },
            }, SAVE_PATH)

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}  |  Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved → {SAVE_PATH}")
    return model

# ── Greedy Decode (inference) ─────────────────────────────────────────────────

def greedy_decode(model, src_text, src_lang, tgt_lang, max_len=MAX_LEN):
    """Translate a single sentence."""
    model.eval()
    src_ids = encode_sentence(src_text, src_lang)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

    tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
    tgt_ids = [BOS_ID, tgt_tag_id]

    with torch.no_grad():
        for _ in range(max_len):
            tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
            logits = model(src_tensor, tgt_tensor)
            next_id = logits[0, -1].argmax().item()
            if next_id == EOS_ID:
                break
            tgt_ids.append(next_id)

    return decode_sentence(tgt_ids, tgt_lang)

# ── Interactive Test ─────────────────────────────────────────────────────────

def interactive():
    print("Loading model …")
    ckpt = torch.load(SAVE_PATH, map_location=DEVICE, weights_only=False)

    # Check checkpoint compatibility with current vocabulary
    try:
        check_checkpoint_compatibility(ckpt["vocab_size"])
    except ValueError as e:
        print(f"Error: {e}")
        print("Aborting checkpoint loading.")
        return

    model = Translator(
        ckpt["vocab_size"], **ckpt["config"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("Model loaded. Enter sentences to translate (Ctrl+C to quit).\n")

    lang_names = {"am": "Amharic", "or": "Afan Oromo", "en": "English"}
    while True:
        try:
            print("Source language (am/or/en): ", end="")
            src_lang = input().strip().lower()
            print("Target language (am/or/en): ", end="")
            tgt_lang = input().strip().lower()
            if src_lang not in SP or tgt_lang not in SP:
                print("Invalid language code. Use am, or, en.\n")
                continue
            print(f"Enter {lang_names[src_lang]} text: ", end="")
            text = input().strip()
            result = greedy_decode(model, text, src_lang, tgt_lang)
            print(f"→ {lang_names[tgt_lang]}: {result}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--test" in sys.argv:
        interactive()
    else:
        model = train()
        # Demo translations after training
        print("\n" + "=" * 60)
        print("DEMO TRANSLATIONS")
        print("=" * 60)
        demos = [
            ("I can't stand this noise anymore.", "en", "am"),
            ("I can't stand this noise anymore.", "en", "or"),
            ("ቃላት ሊገልፁት አይችሉም።", "am", "en"),
            ("ቃላት ሊገልፁት አይችሉም።", "am", "or"),
            ("Jechoonni ibsuu hin danda'an.", "or", "en"),
            ("Jechoonni ibsuu hin danda'an.", "or", "am"),
        ]
        for text, src, tgt in demos:
            result = greedy_decode(model, text, src, tgt)
            print(f"[{src}→{tgt}] {text}")
            print(f"       → {result}\n")
        sys.exit(0)