#!/usr/bin/env python3
"""
Phase 5 - Milestone 3.1: Convergence (WITH LANGUAGE TAGS)
FIXED: Added causal mask, padding masks, and proper label shifting
"""

# ============ TENSORBOARD USAGE ============                          # TB: new
# To launch TensorBoard, run:                                          # TB: new
#   tensorboard --logdir=output/phase_5_milestone_3_1/tensorboard      # TB: new
# Then open http://localhost:6006 in your browser.                     # TB: new
# ============================================                         # TB: new

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter  # TB: new
from collections import defaultdict  # TB: new — for per-pair BLEU tracking

CONFIG = {
    'checkpoint_path': 'output/phase_5_milestone_2_2/trilingual_translator_v1_best.pt',
    'trilingual_path': 'output/data_final/clean_translation_final.csv',
    'en_am_path': 'output/data_final/amharic_english_final.csv',
    'am_or_path': 'output/data_final/amharic_oromo_final.csv',
    'output_dir': 'output/phase_5_milestone_3_1',
    'start_epoch': 41,
    'epochs': 20,
    'batch_size': 64,
    'base_lr': 3e-4,
    'lr_decay_factor': 0.1,
    'lr_decay_every': 5,
    'epoch_size': 80000,
    'tri_pct': 0.50,
    'en_am_pct': 0.25,
    'am_or_pct': 0.25,
    'tri_duplication': 5,
    'tri_weight': 2.0,
    'bi_weight': 0.8,
    'patience': 10,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'max_seq_length': 50,
    'tokenizer_path': 'output/spm_unified_multilingual.model',
    'val_bleu_interval': 2,      # TB: new — evaluate BLEU every N epochs
    'val_bleu_samples': 200,     # TB: new — number of validation samples for BLEU
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# TB: new — Initialize TensorBoard writer
tb_log_dir = os.path.join(CONFIG['output_dir'], 'tensorboard')  # TB: new
writer = SummaryWriter(log_dir=tb_log_dir)                       # TB: new
print(f"📊 TensorBoard log dir: {tb_log_dir}")                   # TB: new
print(f"   Launch with: tensorboard --logdir={tb_log_dir}")       # TB: new

import sentencepiece as spm

print("="*80)
print("PHASE 5 - MILESTONE 3.1: CONVERGENCE (WITH LANGUAGE TAGS) [FIXED]")
print("="*80)

# ============ TOKENIZER & VOCAB ============
print("\n[1/8] Loading tokenizer and building vocabulary...")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(CONFIG['tokenizer_path'])
sp_vocab_size = tokenizer.GetPieceSize()

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

TOKEN2ID = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
next_id = 4

for tag in LANG_TAGS.values():
    TOKEN2ID[tag] = next_id
    next_id += 1

for i in range(sp_vocab_size):
    piece = tokenizer.id_to_piece(i)
    if piece in TOKEN2ID:
        continue
    TOKEN2ID[piece] = next_id
    next_id += 1

ID2TOKEN = {v: k for k, v in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKEN2ID)

print(f"✓ Vocabulary: {VOCAB_SIZE} tokens")

def encode_sentence(text, lang, max_len=CONFIG['max_seq_length']):
    pieces = tokenizer.encode(str(text), out_type=str)
    ids = [TOKEN2ID.get(p, UNK_ID) for p in pieces]
    ids = ids[:max_len - 2]  # Truncate to leave room for BOS and EOS
    ids = [BOS_ID] + ids + [EOS_ID]  # Add BOS and EOS
    
    while len(ids) < max_len:
        ids.append(PAD_ID)
    
    return ids[:max_len]  # Ensure exactly max_len

# ============ DATA LOADING & TRAIN/VAL/TEST SPLIT ============
print("\n[2/8] Loading datasets and creating 80/10/10 train/val/test split...")

# Load raw data
tri_raw = pd.read_csv(CONFIG['trilingual_path'])
en_am_raw = pd.read_csv(CONFIG['en_am_path'])
am_or_raw = pd.read_csv(CONFIG['am_or_path'])

print(f"\n  Original dataset sizes:")
print(f"    Trilingual: {len(tri_raw)} rows")
print(f"    EN-AM:      {len(en_am_raw)} rows")
print(f"    AM-OR:      {len(am_or_raw)} rows")

# Create 80/10/10 splits (random_state=42 for reproducibility)
print(f"\n  Creating 80/10/10 train/val/test splits...")

tri_val_test = tri_raw.sample(frac=0.2, random_state=42)
tri_train = tri_raw.drop(tri_val_test.index)
tri_val = tri_val_test.sample(frac=0.5, random_state=42)
tri_test = tri_val_test.drop(tri_val.index)

en_am_val_test = en_am_raw.sample(frac=0.2, random_state=42)
en_am_train = en_am_raw.drop(en_am_val_test.index)
en_am_val = en_am_val_test.sample(frac=0.5, random_state=42)
en_am_test = en_am_val_test.drop(en_am_val.index)

am_or_val_test = am_or_raw.sample(frac=0.2, random_state=42)
am_or_train = am_or_raw.drop(am_or_val_test.index)
am_or_val = am_or_val_test.sample(frac=0.5, random_state=42)
am_or_test = am_or_val_test.drop(am_or_val.index)

# Save all splits to disk
split_dir = CONFIG['output_dir']
print(f"\n  Saving all splits to {split_dir}/...")

tri_train.to_csv(f'{split_dir}/train_trilingual.csv', index=False)
tri_val.to_csv(f'{split_dir}/val_trilingual.csv', index=False)
tri_test.to_csv(f'{split_dir}/test_trilingual.csv', index=False)

en_am_train.to_csv(f'{split_dir}/train_en_am.csv', index=False)
en_am_val.to_csv(f'{split_dir}/val_en_am.csv', index=False)
en_am_test.to_csv(f'{split_dir}/test_en_am.csv', index=False)

am_or_train.to_csv(f'{split_dir}/train_am_or.csv', index=False)
am_or_val.to_csv(f'{split_dir}/val_am_or.csv', index=False)
am_or_test.to_csv(f'{split_dir}/test_am_or.csv', index=False)

print(f"  ✓ All 9 split files saved")

# Print split breakdown
print(f"\n  Split breakdown:")
print(f"    Trilingual: train={len(tri_train)}, val={len(tri_val)}, test={len(tri_test)}")
print(f"    EN-AM:      train={len(en_am_train)}, val={len(en_am_val)}, test={len(en_am_test)}")
print(f"    AM-OR:      train={len(am_or_train)}, val={len(am_or_val)}, test={len(am_or_test)}")

# Use ONLY train splits for training (duplicate trilingual)
tri_dup = pd.concat([tri_train] * CONFIG['tri_duplication'], ignore_index=True)

print(f"\n  Training set sizes (after duplication):")
print(f"✓ Trilingual x{CONFIG['tri_duplication']}: {len(tri_dup)} rows")
print(f"✓ EN-AM: {len(en_am_train)} rows")
print(f"✓ AM-OR: {len(am_or_train)} rows")

# ============ DATASET ============
class ConvergenceDataset(Dataset):
    def __init__(self, tri_data, en_am_data, am_or_data):
        self.tri_data = tri_data.reset_index(drop=True)
        self.en_am_data = en_am_data.reset_index(drop=True)
        self.am_or_data = am_or_data.reset_index(drop=True)
        
        print("  Pre-tokenizing...")
        
        self.tri_samples = []
        for _, row in self.tri_data.iterrows():
            en, am, or_ = row['English'], row['Amharic'], row['Oromo']
            self.tri_samples.append(('en', 'am', en, am))
            self.tri_samples.append(('am', 'en', am, en))
            self.tri_samples.append(('am', 'or', am, or_))
            self.tri_samples.append(('or', 'am', or_, am))
            self.tri_samples.append(('en', 'or', en, or_))
            self.tri_samples.append(('or', 'en', or_, en))
        
        self.enam_samples = []
        for _, row in self.en_am_data.iterrows():
            self.enam_samples.append(('en', 'am', row['English'], row['Amharic']))
            self.enam_samples.append(('am', 'en', row['Amharic'], row['English']))
        
        self.amor_samples = []
        for _, row in self.am_or_data.iterrows():
            c = list(row)
            self.amor_samples.append(('am', 'or', c[0], c[1]))
            self.amor_samples.append(('or', 'am', c[1], c[0]))
        
        print("  ✓ Pre-tokenization complete")
    
    def __len__(self):
        return CONFIG['epoch_size']
    
    def __getitem__(self, idx):
        r = np.random.random()
        
        if r < CONFIG['tri_pct']:
            s = self.tri_samples[np.random.randint(0, len(self.tri_samples))]
            pair_type = 'trilingual'
            weight = CONFIG['tri_weight']
        elif r < (CONFIG['tri_pct'] + CONFIG['en_am_pct']):
            s = self.enam_samples[np.random.randint(0, len(self.enam_samples))]
            pair_type = 'en_am'
            weight = CONFIG['bi_weight']
        else:
            s = self.amor_samples[np.random.randint(0, len(self.amor_samples))]
            pair_type = 'am_or'
            weight = CONFIG['bi_weight']
        
        src_lang, tgt_lang, src_text, tgt_text = s
        
        src_ids = encode_sentence(src_text, src_lang)
        tgt_ids = encode_sentence(tgt_text, tgt_lang)
        tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
        # FIX 1: Prepend language tag AFTER BOS, don't replace existing BOS
        # Original was: tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]
        # This replaced the first token (BOS) with tag — WRONG
        # Fixed: insert tag right after BOS, keep rest of sequence
        tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:-1] + [EOS_ID]
        # Ensure padding is correct after modification
        while len(tgt_ids) < CONFIG['max_seq_length']:
            tgt_ids.append(PAD_ID)
        tgt_ids = tgt_ids[:CONFIG['max_seq_length']]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'pair_type': pair_type,
            'loss_weight': weight,
        }

dataset = ConvergenceDataset(tri_dup, en_am_train, am_or_train)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=2, pin_memory=False)

print(f"✓ Dataset: {len(dataset)} samples/epoch")

# ============ MODEL ============
# FIX 2: Added causal mask and padding mask support to forward()
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=512, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    # FIX 2a: forward() now accepts masks
    def forward(self, src, tgt, tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        # FIX 2b: Pass padding mask to encoder
        enc_out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        # FIX 2c: Pass causal mask AND padding mask to decoder
        dec_out = self.decoder(
            tgt_emb, enc_out,
            tgt_mask=tgt_mask,                    # causal mask (upper triangular)
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.output_projection(dec_out)
    
    def get_encoder_params(self):
        return list(self.encoder.parameters())
    
    def get_decoder_params(self):
        return [p for n, p in self.named_parameters() if 'encoder.' not in n]

# ============ LOAD CHECKPOINT ============
print("\n[4/8] Loading checkpoint from Phase 5.2...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTransformer(VOCAB_SIZE, CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'])

start_epoch_abs = CONFIG['start_epoch']
best_val_bleu = 0.0
patience_counter = 0

if os.path.exists(CONFIG['checkpoint_path']):
    ckpt = torch.load(CONFIG['checkpoint_path'], map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print("✓ Loaded from Phase 5.2")
        if 'epoch' in ckpt:
            start_epoch_abs = ckpt['epoch'] + 1
    else:
        print("⚠ Could not load checkpoint")
else:
    print("⚠ No checkpoint found")

model = model.to(device)

# ============ TRAINING ============
print("\n[5/8] Starting convergence training...")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
losses = []
val_bleus = []
lr_history = []
best_epoch = 0
best_loss = float('inf')

start_time = time.time()

# FIX 3: Helper function to generate causal mask
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()  # True = masked position

# TB: new — Greedy decode for BLEU evaluation
@torch.no_grad()
def greedy_decode(model, src_tensor, tgt_lang, max_len=CONFIG['max_seq_length']):
    """Autoregressively generate target tokens given a source tensor."""
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)  # (1, seq_len)
    src_pad_mask = (src == PAD_ID)
    
    tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
    generated = [BOS_ID, tgt_tag_id]
    
    for _ in range(max_len - 2):
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_len = tgt_tensor.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        tgt_pad_mask = (tgt_tensor == PAD_ID)
        
        logits = model(src, tgt_tensor,
                       tgt_mask=tgt_mask,
                       src_key_padding_mask=src_pad_mask,
                       tgt_key_padding_mask=tgt_pad_mask)
        next_token = logits[0, -1, :].argmax().item()
        
        if next_token == EOS_ID:
            break
        generated.append(next_token)
    
    return generated[2:]  # Strip BOS and language tag

# TB: new — BLEU-4 computation (corpus-level)
def compute_bleu(references, hypotheses, max_n=4):
    """Compute corpus-level BLEU score with brevity penalty."""
    from collections import Counter
    import math
    
    clipped_counts = [0] * max_n
    total_counts = [0] * max_n
    ref_len = 0
    hyp_len = 0
    
    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        
        for n in range(1, max_n + 1):
            ref_ngrams = Counter()
            for i in range(len(ref) - n + 1):
                ref_ngrams[tuple(ref[i:i+n])] += 1
            
            hyp_ngrams = Counter()
            for i in range(len(hyp) - n + 1):
                hyp_ngrams[tuple(hyp[i:i+n])] += 1
            
            for ng, count in hyp_ngrams.items():
                clipped_counts[n-1] += min(count, ref_ngrams.get(ng, 0))
                total_counts[n-1] += count
    
    # Avoid log(0)
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / total_counts[n])
    
    # Geometric mean of precisions
    log_avg = sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / max_n
    if log_avg == float('-inf'):
        return 0.0
    
    # Brevity penalty
    if hyp_len == 0:
        return 0.0
    bp = math.exp(min(0, 1 - ref_len / hyp_len))
    
    return bp * math.exp(log_avg)

# TB: new — Evaluate BLEU on validation splits
@torch.no_grad()
def evaluate_bleu(model, tri_val, en_am_val, am_or_val, n_samples=CONFIG['val_bleu_samples']):
    """Compute BLEU on a sample of validation data for each language pair."""
    model.eval()
    pair_results = defaultdict(lambda: {'refs': [], 'hyps': []})
    
    # --- Trilingual validation pairs ---
    tri_sample = tri_val.sample(n=min(n_samples // 4, len(tri_val)), random_state=None)
    for _, row in tri_sample.iterrows():
        for src_lang, tgt_lang, src_col, tgt_col in [
            ('en', 'am', 'English', 'Amharic'),
            ('am', 'en', 'Amharic', 'English'),
        ]:
            src_ids = encode_sentence(row[src_col], src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = greedy_decode(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(row[tgt_col], tgt_lang)
            # Strip special tokens from reference
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            # Also strip any language tag ids from ref
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    # --- EN-AM validation pairs ---
    enam_sample = en_am_val.sample(n=min(n_samples // 4, len(en_am_val)), random_state=None)
    for _, row in enam_sample.iterrows():
        for src_lang, tgt_lang, src_col, tgt_col in [
            ('en', 'am', 'English', 'Amharic'),
            ('am', 'en', 'Amharic', 'English'),
        ]:
            src_ids = encode_sentence(row[src_col], src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = greedy_decode(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(row[tgt_col], tgt_lang)
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    # --- AM-OR validation pairs ---
    amor_sample = am_or_val.sample(n=min(n_samples // 4, len(am_or_val)), random_state=None)
    amor_cols = list(amor_sample.columns)
    for _, row in amor_sample.iterrows():
        c = list(row)
        for src_lang, tgt_lang, src_text, tgt_text in [
            ('am', 'or', c[0], c[1]),
            ('or', 'am', c[1], c[0]),
        ]:
            src_ids = encode_sentence(src_text, src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = greedy_decode(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(tgt_text, tgt_lang)
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    # Compute BLEU per pair
    bleu_scores = {}
    for pair_key, data in sorted(pair_results.items()):
        bleu_scores[pair_key] = compute_bleu(data['refs'], data['hyps'])
    
    # Overall average
    if bleu_scores:
        bleu_scores['AVG'] = sum(bleu_scores.values()) / len(bleu_scores)
    else:
        bleu_scores['AVG'] = 0.0
    
    model.train()
    return bleu_scores

for epoch_idx in range(CONFIG['epochs']):
    abs_epoch = CONFIG['start_epoch'] + epoch_idx
    
    decay_steps = epoch_idx // CONFIG['lr_decay_every']
    current_lr = CONFIG['base_lr'] * (CONFIG['lr_decay_factor'] ** decay_steps)
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    
    model.train()
    epoch_loss = 0
    batch_count = 0
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        weights = batch['loss_weight'].to(device)
        
        optimizer.zero_grad()
        
        # FIX 4: Label shifting is already correct here
        # tgt_input = everything except last token (what decoder sees)
        # tgt_output = everything except first token (what we predict)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # FIX 5: Create masks
        # 5a: Causal mask for decoder (prevents looking at future tokens)
        tgt_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        
        # 5b: Padding masks (tell model to ignore pad tokens)
        src_key_padding_mask = (src == PAD_ID)  # True where pad
        tgt_key_padding_mask = (tgt_input == PAD_ID)
        
        # FIX 6: Pass all masks to model
        logits = model(
            src, tgt_input,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss_per_token = criterion(logits, tgt_output)
        batch_weight = weights.mean()
        loss = loss_per_token * batch_weight
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # TB: new — Log batch loss
        global_step = epoch_idx * len(dataloader) + batch_idx  # TB: new
        writer.add_scalar('Train/Loss_batch', loss.item(), global_step)  # TB: new
        
        if (batch_idx + 1) % 500 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            writer.add_scalar('Train/LR', current_lr, global_step)  # TB: new — LR every 500 batches
    
    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    lr_history.append(current_lr)
    
    # TB: new — Log epoch-level metrics
    epoch_global_step = (epoch_idx + 1) * len(dataloader)              # TB: new
    writer.add_scalar('Train/Loss_epoch', avg_loss, epoch_global_step)  # TB: new
    writer.add_scalar('Train/LR', current_lr, epoch_global_step)        # TB: new — LR at epoch end
    
    # TB: new — Real BLEU evaluation on validation data
    val_bleu = 0.0
    bleu_str = ''
    if (epoch_idx + 1) % CONFIG['val_bleu_interval'] == 0 or epoch_idx == CONFIG['epochs'] - 1:  # TB: new
        print(f"    Evaluating BLEU on validation set...")                                       # TB: new
        bleu_scores = evaluate_bleu(model, tri_val, en_am_val, am_or_val)                        # TB: new
        val_bleu = bleu_scores.get('AVG', 0.0)                                                   # TB: new
        # Log per-pair BLEU to TensorBoard                                                       # TB: new
        for pair_key, score in bleu_scores.items():                                              # TB: new
            tag = f"Val/BLEU_{pair_key.replace('→', '_to_')}"                                    # TB: new
            writer.add_scalar(tag, score, epoch_global_step)                                     # TB: new
        # Print per-pair breakdown                                                               # TB: new
        pair_strs = [f"{k}={v:.4f}" for k, v in bleu_scores.items()]                             # TB: new
        print(f"    BLEU: {' | '.join(pair_strs)}")                                              # TB: new
        bleu_str = f" | BLEU={val_bleu:.4f}"                                                     # TB: new
    val_bleus.append(val_bleu)
    
    # Save best by loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_epoch = abs_epoch
        patience_counter = 0
        torch.save({
            'epoch': abs_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'val_bleu': val_bleu,
            'config': CONFIG,
            'token2id': TOKEN2ID,
            'id2token': ID2TOKEN,
            'vocab_size': VOCAB_SIZE,
        }, f"{CONFIG['output_dir']}/final_translator_best.pt")
        print(f"    ✓ NEW BEST saved (loss: {avg_loss:.4f})")
        writer.add_scalar('Train/Best_loss', avg_loss, epoch_global_step)  # TB: new — Best loss marker
    else:
        patience_counter += 1
    
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    
    def fmt(t):
        return f"{int(t//3600)}h {int((t%3600)//60)}m"
    
    print(f"  Epoch {abs_epoch:2d}: Loss={avg_loss:.4f} | LR={current_lr:.0e}{bleu_str}"
          f" | Time={fmt(epoch_time)} | Elapsed={fmt(elapsed)}{' [FROZEN]' if False else ''}")
    
    if patience_counter >= CONFIG['patience']:
        print(f"\n  Early stopping!")
        break

# Final save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'final_epoch': best_epoch,
    'final_loss': losses[-1],
    'token2id': TOKEN2ID,
    'id2token': ID2TOKEN,
    'vocab_size': VOCAB_SIZE,
    'training_history': {
        'losses': [float(l) for l in losses],
        'val_bleus': val_bleus,
        'lr_history': lr_history,
    }
}, f"{CONFIG['output_dir']}/final_translator_multilingual.pt")

writer.close()  # TB: new — Flush and close TensorBoard writer
print(f"📊 TensorBoard logs saved to: {tb_log_dir}")  # TB: new

print("\n" + "="*80)
print("MILESTONE 3.1 COMPLETE")
print("="*80)
print(f"✓ Final model: {CONFIG['output_dir']}/final_translator_multilingual.pt")
print(f"✓ Best model:  {CONFIG['output_dir']}/final_translator_best.pt")
print(f"✓ Vocab size: {VOCAB_SIZE} (with language tags)")
print("="*80)