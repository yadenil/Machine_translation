"""
Milestone 1.3: Baseline Training (Bilingual Foundation)
========================================================
Trains a multilingual translator on:
- 50% Amharic-English pairs (209,837 samples)
- 50% Amharic-Oromo pairs (143,987 samples)

Phase 1: Bilingual Foundation (Epochs 1-15)
- Objective: Build strong encoder representations
- No trilingual data yet (held for validation)
- Equal sampling of both bilingual pairs

Usage:
    python train_baseline_bilingual.py          # train the model
    python train_baseline_bilingual.py --test   # interactive inference
"""

import os, sys, math, random, json
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from datetime import datetime

try:
    import sacrebleu
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False
    print("⚠ Warning: sacrebleu not installed. Install with: pip install sacrebleu")

try:
    from torchtext.data.metrics import bleu_score
    TORCHTEXT_AVAILABLE = True
except ImportError:
    TORCHTEXT_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Milestone 1.3 specific paths
BASE_PATH = "output"
DATA_FINAL_PATH = os.path.join(BASE_PATH, "data_final")
CHECKPOINT_DIR = os.path.join(BASE_PATH, "milestone_1_3")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Data files (from Milestone 1.1)
BILINGUAL_EN_AM_PATH = os.path.join(DATA_FINAL_PATH, "amharic_english_final.csv")
BILINGUAL_AM_OR_PATH = os.path.join(DATA_FINAL_PATH, "amharic_oromo_final.csv")
VALIDATION_DATA_PATH = os.path.join(BASE_PATH, "cleaned_dataset", "clean_translation.csv")

# Model hyperparameters
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
MAX_LEN = 80

# Training hyperparameters (Milestone 1.3)
EPOCHS = 15  # Phase 1: Bilingual foundation only
BATCH_SIZE = 64  # Doubled for larger dataset
LR = 3e-4
LABEL_SMOOTHING = 0.1

# Special token IDs
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# Language tags
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

# ── Load Unified Tokenizer ───────────────────────────────────────────────────
def load_sp(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

unified_tokenizer_path = os.path.join(BASE_PATH, "spm_unified_multilingual.model")

if os.path.isfile(unified_tokenizer_path):
    print(f"✓ Loading unified multilingual tokenizer from {unified_tokenizer_path}")
    sp_unified = load_sp(unified_tokenizer_path)
    SP = {"am": sp_unified, "or": sp_unified, "en": sp_unified}
else:
    raise FileNotFoundError(f"Tokenizer not found at {unified_tokenizer_path}")

# ── Build vocabulary from unified tokenizer ───────────────────────────────────
def build_unified_vocab():
    """Build vocabulary from the unified multilingual SentencePiece tokenizer."""
    token2id = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
    next_id = 4

    # Add language tags
    for tag in LANG_TAGS.values():
        token2id[tag] = next_id
        next_id += 1

    # Add tokens from the unified tokenizer
    sp = SP["am"]
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if piece in token2id:
            continue
        token2id[piece] = next_id
        next_id += 1

    # Reverse mapping
    id2token = {v: k for k, v in token2id.items()}

    print("\n" + "=" * 70)
    print("VOCABULARY STATISTICS (UNIFIED MULTILINGUAL)")
    print("=" * 70)
    print(f"Tokenizer: spm_unified_multilingual")
    print(f"Base vocabulary: {sp.get_piece_size():,} tokens")
    print(f"Plus language tags: {len(LANG_TAGS)}")
    print(f"Plus special tokens: 4")
    print(f"Total unified vocabulary: {len(token2id):,} tokens")
    print("=" * 70 + "\n")

    return token2id, id2token

TOKEN2ID, ID2TOKEN = build_unified_vocab()
VOCAB_SIZE = len(TOKEN2ID)

# ── Encode / Decode helpers ──────────────────────────────────────────────────
def encode_sentence(text, lang, max_len=MAX_LEN):
    """Tokenize text using the unified multilingual SentencePiece tokenizer."""
    sp = SP[lang]
    pieces = sp.encode(str(text), out_type=str)
    ids = [TOKEN2ID.get(p, UNK_ID) for p in pieces]
    ids = ids[: max_len - 2]
    return [BOS_ID] + ids + [EOS_ID]

def decode_sentence(ids, lang):
    """Convert unified IDs back to text using the language-specific SPM."""
    sp = SP[lang]
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

# ── Dataset ──────────────────────────────────────────────────────────────────
class BilingualTranslationDataset(Dataset):
    """
    Creates translation pairs from bilingual data.
    For EN-AM data: creates both EN→AM and AM→EN directions
    For AM-OR data: creates both AM→OR and OR→AM directions
    """
    def __init__(self, en_am_df=None, am_or_df=None, src_lang=None, tgt_lang=None, max_len=MAX_LEN):
        self.samples = []
        
        if src_lang == "en" and tgt_lang == "am" and en_am_df is not None:
            # EN → AM
            for _, row in en_am_df.iterrows():
                src_ids = encode_sentence(row['English'], 'en', max_len)
                tgt_ids = encode_sentence(row['Amharic'], 'am', max_len)
                tgt_tag_id = TOKEN2ID[LANG_TAGS['am']]
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
                self.samples.append((src_ids, tgt_ids))
                
        elif src_lang == "am" and tgt_lang == "en" and en_am_df is not None:
            # AM → EN
            for _, row in en_am_df.iterrows():
                src_ids = encode_sentence(row['Amharic'], 'am', max_len)
                tgt_ids = encode_sentence(row['English'], 'en', max_len)
                tgt_tag_id = TOKEN2ID[LANG_TAGS['en']]
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
                self.samples.append((src_ids, tgt_ids))
                
        elif src_lang == "am" and tgt_lang == "or" and am_or_df is not None:
            # AM → OR
            for _, row in am_or_df.iterrows():
                src_ids = encode_sentence(row['Amharic'], 'am', max_len)
                tgt_ids = encode_sentence(row['Oromo'], 'or', max_len)
                tgt_tag_id = TOKEN2ID[LANG_TAGS['or']]
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
                self.samples.append((src_ids, tgt_ids))
                
        elif src_lang == "or" and tgt_lang == "am" and am_or_df is not None:
            # OR → AM
            for _, row in am_or_df.iterrows():
                src_ids = encode_sentence(row['Oromo'], 'or', max_len)
                tgt_ids = encode_sentence(row['Amharic'], 'am', max_len)
                tgt_tag_id = TOKEN2ID[LANG_TAGS['am']]
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
                self.samples.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt = self.samples[idx]
        return src, tgt

class TrilingualValidationDataset(Dataset):
    """Trilingual validation dataset (all 6 directions)"""
    PAIRS = [("am", "en"), ("en", "am"), ("am", "or"), ("or", "am"), ("en", "or"), ("or", "en")]
    COL = {"am": "Amharic", "or": "Oromo", "en": "English"}

    def __init__(self, df, max_len=MAX_LEN):
        self.samples = []
        for src_lang, tgt_lang in self.PAIRS:
            src_col = self.COL[src_lang]
            tgt_col = self.COL[tgt_lang]
            tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
            for _, row in df.iterrows():
                src_ids = encode_sentence(row[src_col], src_lang, max_len)
                tgt_ids = encode_sentence(row[tgt_col], tgt_lang, max_len)
                tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
                self.samples.append((src_ids, tgt_ids, src_lang, tgt_lang))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, src_lang, tgt_lang = self.samples[idx]
        return src, tgt, src_lang, tgt_lang

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)
    src_padded = torch.tensor([pad_sequence(s, max_src) for s in src_batch], dtype=torch.long)
    tgt_padded = torch.tensor([pad_sequence(t, max_tgt) for t in tgt_batch], dtype=torch.long)
    return src_padded, tgt_padded

def collate_val_fn(batch):
    src_batch, tgt_batch, src_langs, tgt_langs = zip(*batch)
    max_src = max(len(s) for s in src_batch)
    max_tgt = max(len(t) for t in tgt_batch)
    src_padded = torch.tensor([pad_sequence(s, max_src) for s in src_batch], dtype=torch.long)
    tgt_padded = torch.tensor([pad_sequence(t, max_tgt) for t in tgt_batch], dtype=torch.long)
    return src_padded, tgt_padded, src_langs, tgt_langs

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
        self.register_buffer("pe", pe.unsqueeze(0))

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
        return (x == PAD_ID)

    def make_causal_mask(self, sz):
        return torch.triu(torch.ones(sz, sz, device=DEVICE), diagonal=1).bool()

    def forward(self, src, tgt):
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

# ── BLEU Evaluation ──────────────────────────────────────────────────────────
def compute_corpus_bleu(hypotheses, references):
    """Compute corpus-level BLEU using sacrebleu."""
    if not SACREBLEU_AVAILABLE or len(hypotheses) == 0:
        return None

    try:
        bleu = sacrebleu.BLEU()
        score = bleu.corpus_score(hypotheses, [references])
        return score.score
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        return None

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

# ── Training ─────────────────────────────────────────────────────────────────
def train():
    print(f"\n{'='*80}")
    print("MILESTONE 1.3: BASELINE TRAINING (BILINGUAL FOUNDATION)")
    print(f"{'='*80}\n")
    
    # Load bilingual data
    print("Loading bilingual datasets...")
    df_en_am = pd.read_csv(BILINGUAL_EN_AM_PATH)
    df_am_or = pd.read_csv(BILINGUAL_AM_OR_PATH)
    print(f"  EN-AM pairs: {len(df_en_am):,}")
    print(f"  AM-OR pairs: {len(df_am_or):,}")
    print(f"  Total bilingual: {len(df_en_am) + len(df_am_or):,}")

    # Load validation data (trilingual)
    print(f"\nLoading validation data (trilingual)...")
    df_val_trilingual = pd.read_csv(VALIDATION_DATA_PATH, sep="\t")
    df_val_trilingual.columns = ["Amharic", "Oromo", "English"]
    df_val_trilingual = df_val_trilingual.dropna().reset_index(drop=True)
    print(f"  Validation triples: {len(df_val_trilingual)}")

    # Create bilingual datasets (50% EN-AM, 50% AM-OR)
    # Create 4 direction-specific datasets
    print(f"\nCreating bilingual datasets (4 directions)...")
    
    ds_en_am = BilingualTranslationDataset(en_am_df=df_en_am, src_lang="en", tgt_lang="am")
    ds_am_en = BilingualTranslationDataset(en_am_df=df_en_am, src_lang="am", tgt_lang="en")
    ds_am_or = BilingualTranslationDataset(am_or_df=df_am_or, src_lang="am", tgt_lang="or")
    ds_or_am = BilingualTranslationDataset(am_or_df=df_am_or, src_lang="or", tgt_lang="am")
    
    # Create validation dataset
    val_ds = TrilingualValidationDataset(df_val_trilingual)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_val_fn)

    # Combine all training datasets
    combined_dataset = torch.utils.data.ConcatDataset([ds_en_am, ds_am_en, ds_am_or, ds_or_am])
    train_dl = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print(f"  EN→AM samples: {len(ds_en_am)}")
    print(f"  AM→EN samples: {len(ds_am_en)}")
    print(f"  AM→OR samples: {len(ds_am_or)}")
    print(f"  OR→AM samples: {len(ds_or_am)}")
    print(f"  Total training samples: {len(combined_dataset):,}")
    print(f"  Training batches per epoch: {len(train_dl)}")

    # Model
    print(f"\nInitializing model...")
    model = Translator(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, MAX_LEN).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")
    print(f"  Device: {DEVICE}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    # Training loop
    print(f"\n{'='*80}")
    print(f"PHASE 1: BILINGUAL FOUNDATION (Epochs 1-{EPOCHS})")
    print(f"{'='*80}\n")

    best_val_loss = float("inf")
    best_bleu = {}
    training_history = []

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (src, tgt) in enumerate(train_dl):
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
            
            # Calculate token-level accuracy
            predictions = logits.argmax(dim=-1)
            mask = tgt_out != PAD_ID
            train_correct += ((predictions == tgt_out) & mask).sum().item()
            train_total += mask.sum().item()

        scheduler.step()
        train_loss /= len(train_dl)
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for src, tgt, _, _ in val_dl:
                src, tgt = src.to(DEVICE), tgt.to(DEVICE)
                tgt_in = tgt[:, :-1]
                tgt_out = tgt[:, 1:]
                logits = model(src, tgt_in)
                loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
                val_loss += loss.item()

                predictions = logits.argmax(dim=-1)
                mask = tgt_out != PAD_ID
                val_correct += ((predictions == tgt_out) & mask).sum().item()
                val_total += mask.sum().item()

        val_loss /= len(val_dl)
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        # ── BLEU Evaluation ──
        bleu_scores = {}
        if SACREBLEU_AVAILABLE and (epoch % 3 == 0 or epoch == 1):
            print(f"Epoch {epoch:2d}/{EPOCHS}  Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

            language_pairs = [("en", "am"), ("am", "en"), ("am", "or"), ("or", "am"), ("en", "or"), ("or", "en")]
            bleu_text = "  BLEU scores: "
            
            # Generate translations for validation set
            for src_lang, tgt_lang in language_pairs:
                hypotheses = []
                references = []
                
                with torch.no_grad():
                    for _, row in df_val_trilingual.iterrows():
                        col_map = {"am": "Amharic", "or": "Oromo", "en": "English"}
                        src_text = str(row[col_map[src_lang]])
                        ref_text = str(row[col_map[tgt_lang]])
                        
                        try:
                            hyp = greedy_decode(model, src_text, src_lang, tgt_lang)
                            hypotheses.append(hyp)
                            references.append(ref_text)
                        except Exception:
                            continue
                
                if len(hypotheses) > 0:
                    bleu = compute_corpus_bleu(hypotheses, references)
                    if bleu is not None:
                        bleu_scores[f"{src_lang}→{tgt_lang}"] = bleu
                        bleu_text += f"{src_lang}→{tgt_lang}:{bleu:.1f}  "
            
            print(bleu_text)
        else:
            print(f"Epoch {epoch:2d}/{EPOCHS}  Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
                  f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_bleu = bleu_scores.copy()
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_translator_bilingual.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "vocab_size": VOCAB_SIZE,
                "token2id": TOKEN2ID,
                "id2token": ID2TOKEN,
                "config": {
                    "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
                    "d_ff": D_FF, "dropout": DROPOUT, "max_len": MAX_LEN,
                },
                "bleu_scores": bleu_scores,
            }, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Record history
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "bleu_scores": bleu_scores,
        })

    # Final evaluation
    print(f"\n{'='*80}")
    print(f"MILESTONE 1.3: BASELINE TRAINING COMPLETE")
    print(f"{'='*80}\n")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best BLEU scores: {best_bleu}\n")

    # Save final metrics
    metrics_path = os.path.join(CHECKPOINT_DIR, "baseline_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            "training_phase": "Phase 1: Bilingual Foundation",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "best_val_loss": best_val_loss,
            "best_bleu_scores": best_bleu,
            "training_data": {
                "en_am_pairs": len(df_en_am),
                "am_or_pairs": len(df_am_or),
                "total_training_samples": len(combined_dataset),
            },
            "validation_data": {
                "trilingual_triples": len(df_val_trilingual),
            },
            "model_config": {
                "d_model": D_MODEL,
                "n_heads": N_HEADS,
                "n_layers": N_LAYERS,
                "d_ff": D_FF,
                "dropout": DROPOUT,
                "vocab_size": VOCAB_SIZE,
                "total_parameters": total_params,
            },
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"✓ Metrics saved: {metrics_path}")

    # Save training history
    history_path = os.path.join(CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Training history saved: {history_path}")

    print(f"\nModel checkpoint: {os.path.join(CHECKPOINT_DIR, 'baseline_translator_bilingual.pt')}")
    print(f"Ready for: Milestone 1.2 - Tokenizer Retraining (optional)")
    print(f"Ready for: Milestone 2.1 - Trilingual Augmentation\n")

    return model

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if "--test" in sys.argv:
        print("Loading model…")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_translator_bilingual.pt")
        ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

        model = Translator(
            ckpt["vocab_size"], **ckpt["config"]
        ).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print("✓ Model loaded. Enter sentences to translate (Ctrl+C to quit).\n")

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
    else:
        model = train()
