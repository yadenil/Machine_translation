#!/usr/bin/env python3
"""
Phase 5 - Milestone 2.2: Encoder Freezing Experiments (FIXED)
"""

import os
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

# ============ CONFIGURATION ============
CONFIG = {
    'model_path': 'output/translator.pt',
    'augmented_data_path': 'output/phase_5_milestone_2_1/trilingual_augmented_3x.csv',
    'en_am_path': 'output/data_final/amharic_english_final.csv',
    'am_or_path': 'output/data_final/amharic_oromo_final.csv',
    'output_dir': 'output/phase_5_milestone_2_2',
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 25,  # Epochs 16-40
    'freeze_every_n_epochs': 5,
    'freeze_duration': 1,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'max_seq_length': 50,
    'tokenizer_path': 'output/spm_unified_multilingual.model',
    'epoch_size': 80000,  # Total samples per epoch (not just trilingual!)
    'val_every': 3,  # Validate every 3 epochs
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

import sentencepiece as spm

print("="*80)
print("PHASE 5 - MILESTONE 2.2: ENCODER FREEZING EXPERIMENTS (FIXED)")
print("="*80)

# ============ TOKENIZER ============
print("\n[1/7] Loading tokenizer...")
tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(CONFIG['tokenizer_path'])
vocab_size = tokenizer.GetPieceSize()
print(f"✓ Tokenizer loaded: {vocab_size} tokens")

# ============ DATA LOADING ============
print("\n[2/7] Loading training data...")

aug_df = pd.read_csv(CONFIG['augmented_data_path'])
en_am_df = pd.read_csv(CONFIG['en_am_path'])
am_or_df = pd.read_csv(CONFIG['am_or_path'])

print(f"✓ Augmented trilingual: {len(aug_df)} rows")
print(f"✓ EN-AM bilingual: {len(en_am_df)} rows")
print(f"✓ AM-OR bilingual: {len(am_or_df)} rows")

# ============ FIXED DATASET: Proper epoch size with 40/30/30 ============
class MixedTrilingualDataset(Dataset):
    def __init__(self, aug_data, en_am_data, am_or_data, tokenizer, max_len=50):
        self.aug_data = aug_data.reset_index(drop=True)
        self.en_am_data = en_am_data.reset_index(drop=True)
        self.am_or_data = am_or_data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Pre-tokenize everything to speed up training
        print("  Pre-tokenizing datasets (this may take a minute)...")
        self.tri_src = [self._tokenize(row['English']) for _, row in self.aug_data.iterrows()]
        self.tri_tgt = [self._tokenize(row['Amharic']) for _, row in self.aug_data.iterrows()]
        self.tri_src_or = [self._tokenize(row['English']) for _, row in self.aug_data.iterrows()]
        self.tri_tgt_or = [self._tokenize(row['Oromo']) for _, row in self.aug_data.iterrows()]
        
        # EN-AM: column 1 = English, column 0 = Amharic (verify this!)
        self.enam_src = [self._tokenize(row[self.en_am_data.columns[1]]) for _, row in self.en_am_data.iterrows()]
        self.enam_tgt = [self._tokenize(row[self.en_am_data.columns[0]]) for _, row in self.en_am_data.iterrows()]
        
        # AM-OR: column 1 = Amharic, column 0 = Oromo (verify this!)
        self.amor_src = [self._tokenize(row[self.am_or_data.columns[1]]) for _, row in self.am_or_data.iterrows()]
        self.amor_tgt = [self._tokenize(row[self.am_or_data.columns[0]]) for _, row in self.am_or_data.iterrows()]
        
        print("  ✓ Pre-tokenization complete")
    
    def _tokenize(self, text):
        tokens = self.tokenizer.encode(str(text))
        tokens = tokens[:CONFIG['max_seq_length']]
        tokens = tokens + [0] * (CONFIG['max_seq_length'] - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)
    
    def __len__(self):
        return CONFIG['epoch_size']  # FIXED: 80K samples per epoch, not 8K
    
    def __getitem__(self, idx):
        # 40/30/30 sampling with FIXED epoch size
        rand = np.random.random()
        
        if rand < 0.40:  # Trilingual (40%)
            tri_idx = np.random.randint(0, len(self.aug_data))
            # Randomly pick EN-AM or EN-OR direction from trilingual
            if np.random.random() < 0.5:
                src, tgt = self.tri_src[tri_idx], self.tri_tgt[tri_idx]
            else:
                src, tgt = self.tri_src_or[tri_idx], self.tri_tgt_or[tri_idx]
            pair_type = 'trilingual'
            weight = 1.5  # Higher loss weight for trilingual
            
        elif rand < 0.70:  # EN-AM (30%)
            bi_idx = np.random.randint(0, len(self.en_am_data))
            src, tgt = self.enam_src[bi_idx], self.enam_tgt[bi_idx]
            pair_type = 'en_am'
            weight = 1.0
            
        else:  # AM-OR (30%)
            bi_idx = np.random.randint(0, len(self.am_or_data))
            src, tgt = self.amor_src[bi_idx], self.amor_tgt[bi_idx]
            pair_type = 'am_or'
            weight = 1.0
        
        return {
            'src': src,
            'tgt': tgt,
            'pair_type': pair_type,
            'loss_weight': weight
        }

dataset = MixedTrilingualDataset(aug_df, en_am_df, am_or_df, tokenizer)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, 
                        num_workers=2, pin_memory=True)
print(f"✓ Mixed dataset: {len(dataset)} samples per epoch (40% tri, 30% EN-AM, 30% AM-OR)")

# ============ VERIFY SAMPLING ============
print("\n[3/7] Verifying batch composition...")
sample_batches = []
for i, batch in enumerate(dataloader):
    if i >= 10:
        break
    types = batch['pair_type']

counts = Counter(sample_batches)
total = sum(counts.values())
print(f"  Sampled 10 batches ({total} examples):")
for pt, count in counts.items():
    print(f"    {pt}: {count} ({count/total*100:.1f}%)")

# ============ MODEL (same architecture as baseline) ============
class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        enc_out = self.encoder(src_emb)
        dec_out = self.decoder(tgt_emb, enc_out)
        return self.output_projection(dec_out)
    
    def get_encoder_params(self):
        return list(self.encoder.parameters())
    
    def get_decoder_params(self):
        # Everything except encoder
        return [p for n, p in self.named_parameters() if 'encoder.' not in n]

# ============ FIXED MODEL LOADING ============
print("\n[4/7] Loading baseline model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTransformer(vocab_size, CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'])

# FIXED: Handle custom checkpoint format
if os.path.exists(CONFIG['model_path']):
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Loaded from checkpoint['model_state_dict']")
        elif 'model_state' in checkpoint:
            # Your format: custom dict with 'model_state' key
            # Need to map keys or rebuild
            print("⚠ Custom checkpoint format detected")
            print(f"   Keys: {list(checkpoint.keys())}")
            # Try to load state directly if shapes match
            try:
                model.load_state_dict(checkpoint['model_state'])
                print("✓ Loaded from checkpoint['model_state']")
            except:
                print("⚠ Could not auto-load. Attempting manual weight transfer...")
                # Fallback: initialize from scratch (not ideal but prevents crash)
                print("   Using random initialization - TRAIN FROM SCRATCH WARNING")
        else:
            print(f"⚠ Unknown checkpoint keys: {list(checkpoint.keys())}")
            print("   Available keys hint the model structure doesn't match.")
            print("   Please check if your baseline model uses the SAME SimpleTransformer class.")
    else:
        model.load_state_dict(checkpoint)
        print("✓ Loaded checkpoint directly")
else:
    print("⚠ Baseline model not found! Training from scratch.")

model = model.to(device)

# ============ TRAINING WITH PROPER FREEZING ============
print("\n[5/7] Training with periodic encoder freezing...")

criterion = nn.CrossEntropyLoss(ignore_index=0)

losses = []
bleu_scores = []
freeze_log = []


# ============ TRAINING WITH TIME PREDICTION ============
print("\n[5/7] Training with periodic encoder freezing...")

criterion = nn.CrossEntropyLoss(ignore_index=0)
losses = []
freeze_log = []
best_loss = float('inf')

# Time tracking
start_time = time.time()
epoch_times = []

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    # Determine if we freeze this epoch
    should_freeze = (epoch % CONFIG['freeze_every_n_epochs'] == 0)
    
    if should_freeze:
        print(f"\n  >>> EPOCH {epoch+16}: ENCODER FROZEN <<<")
        optimizer = optim.Adam(model.get_decoder_params(), lr=CONFIG['learning_rate'])
        freeze_log.append(f"Epoch {epoch+16}: FROZEN")
    else:
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        freeze_log.append(f"Epoch {epoch+16}: UNFROZEN")
    
    # Training
    model.train()
    epoch_loss = 0
    batch_count = 0
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        weights = batch['loss_weight'].to(device)
        
        optimizer.zero_grad()
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        logits = model(src, tgt_input)
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss_per_token = criterion(logits, tgt_output)
        batch_weight = weights.mean()
        loss = loss_per_token * batch_weight
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        # Print progress every 500 batches
        if (batch_idx + 1) % 500 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    
    # Time tracking
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    elapsed = time.time() - start_time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = CONFIG['epochs'] - epoch - 1
    eta = avg_epoch_time * remaining_epochs
    
    # Format times
    elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
    eta_str = f"{int(eta//3600)}h {int((eta%3600)//60)}m"
    epoch_str = f"{int(epoch_time//60)}m {int(epoch_time%60)}s"
    
    status = " [FROZEN]" if should_freeze else ""
    print(f"  Epoch {epoch+16:2d}: Loss = {avg_loss:.4f} | Time: {epoch_str} | Elapsed: {elapsed_str} | ETA: {eta_str}{status}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 16,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': CONFIG,
        }, f"{CONFIG['output_dir']}/trilingual_translator_v1_best.pt")
        print(f"    ✓ New best model saved (loss: {avg_loss:.4f})")

# Final save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'epoch': 16 + CONFIG['epochs'],
    'final_loss': losses[-1],
}, f"{CONFIG['output_dir']}/trilingual_translator_v1.pt")
# ============ PLOT ============
print("\n[7/7] Generating training curves...")
plt.figure(figsize=(12, 6))
plt.plot(range(16, 16+CONFIG['epochs']), losses, marker='o', linewidth=2, label='Training Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Milestone 2.2: Training with Periodic Encoder Freezing', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Mark frozen epochs
for i, log in enumerate(freeze_log):
    if 'FROZEN' in log:
        plt.axvline(x=16+i, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_curves_trilingual.png", dpi=300)
print(f"✓ Plot saved")

# Report
report = {
    "milestone": "2.2 - Encoder Freezing (FIXED)",
    "freeze_schedule": freeze_log,
    "losses": [float(l) for l in losses],
    "config": CONFIG,
    "verification": {
        "batch_composition": {k: f"{v/total*100:.1f}%" for k, v in counts.items()},
        "epoch_size": len(dataset),
        "trilingual_weight": 1.5,
        "bilingual_weight": 1.0,
    }
}

with open(f"{CONFIG['output_dir']}/milestone_2_2_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("MILESTONE 2.2 COMPLETE")
print("="*80)
print(f"✓ Model saved: {CONFIG['output_dir']}/trilingual_translator_v1.pt")
print(f"✓ Report saved: {CONFIG['output_dir']}/milestone_2_2_report.json")
print(f"✓ Plot saved: {CONFIG['output_dir']}/training_curves_trilingual.png")
print(f"\n⚠ IMPORTANT: You need to add BLEU evaluation code in the validation section.")
print("="*80)