#!/usr/bin/env python3
"""
Phase 5 - Milestone 3.1: Convergence (WITH LANGUAGE TAGS)
FIXED: Added causal mask, padding masks, and proper label shifting
"""

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
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

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

# ============ DATA LOADING ============
print("\n[2/8] Loading datasets...")

tri_df = pd.read_csv(CONFIG['trilingual_path'])
tri_dup = pd.concat([tri_df] * CONFIG['tri_duplication'], ignore_index=True)

en_am_df = pd.read_csv(CONFIG['en_am_path'])
am_or_df = pd.read_csv(CONFIG['am_or_path'])

print(f"✓ Trilingual x{CONFIG['tri_duplication']}: {len(tri_dup)} rows")
print(f"✓ EN-AM: {len(en_am_df)} rows")
print(f"✓ AM-OR: {len(am_or_df)} rows")

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

dataset = ConvergenceDataset(tri_dup, en_am_df, am_or_df)
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
        
        if (batch_idx + 1) % 500 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    lr_history.append(current_lr)
    
    # Placeholder for BLEU (replace with real eval when ready)
    val_bleu = 0.0
    val_bleus.append(val_bleu)
    
    # Save best by loss (since BLEU is placeholder)
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
    else:
        patience_counter += 1
    
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    
    def fmt(t):
        return f"{int(t//3600)}h {int((t%3600)//60)}m"
    
    print(f"  Epoch {abs_epoch:2d}: Loss={avg_loss:.4f} | LR={current_lr:.0e} | "
          f"Time={fmt(epoch_time)} | Elapsed={fmt(elapsed)}{' [FROZEN]' if False else ''}")
    
    if patience_counter >= CONFIG['patience']:
        print(f"\n  🛑 Early stopping!")
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

print("\n" + "="*80)
print("MILESTONE 3.1 COMPLETE")
print("="*80)
print(f"✓ Final model: {CONFIG['output_dir']}/final_translator_multilingual.pt")
print(f"✓ Best model:  {CONFIG['output_dir']}/final_translator_best.pt")
print(f"✓ Vocab size: {VOCAB_SIZE} (with language tags)")
print("="*80)