#!/usr/bin/env python3
"""
Phase 5 - Milestone 2.2: Encoder Freezing Experiments (WITH LANGUAGE TAGS)
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

# ============ CONFIGURATION ============
CONFIG = {
    'model_path': 'output/translator.pt',
    'augmented_data_path': 'output/phase_5_milestone_2_1/trilingual_augmented_3x.csv',
    'en_am_path': 'output/data_final/amharic_english_final.csv',
    'am_or_path': 'output/data_final/amharic_oromo_final.csv',
    'output_dir': 'output/phase_5_milestone_2_2',
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 25,
    'freeze_every_n_epochs': 5,
    'freeze_duration': 1,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 2,
    'max_seq_length': 50,
    'tokenizer_path': 'output/spm_unified_multilingual.model',
    'epoch_size': 80000,
    'val_every': 3,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

import sentencepiece as spm

print("="*80)
print("PHASE 5 - MILESTONE 2.2: ENCODER FREEZING (WITH LANGUAGE TAGS)")
print("="*80)

# ============ TOKENIZER & VOCAB (same as original train_translator.py) ============
print("\n[1/7] Loading tokenizer and building vocabulary...")

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(CONFIG['tokenizer_path'])
sp_vocab_size = tokenizer.GetPieceSize()
print(f"✓ Tokenizer loaded: {sp_vocab_size} tokens")

# Build vocabulary EXACTLY like original train_translator.py
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

TOKEN2ID = {"<pad>": PAD_ID, "<unk>": UNK_ID, "<bos>": BOS_ID, "<eos>": EOS_ID}
next_id = 4

# Add language tags: <am>=4, <or>=5, <en>=6
for tag in LANG_TAGS.values():
    TOKEN2ID[tag] = next_id
    next_id += 1

# Add SentencePiece tokens starting at ID 7
for i in range(sp_vocab_size):
    piece = tokenizer.id_to_piece(i)
    if piece in TOKEN2ID:
        continue
    TOKEN2ID[piece] = next_id
    next_id += 1

ID2TOKEN = {v: k for k, v in TOKEN2ID.items()}
VOCAB_SIZE = len(TOKEN2ID)

print(f"✓ Vocabulary built: {VOCAB_SIZE} tokens")
print(f"  Specials: <pad>=0, <unk>=1, <bos>=2, <eos>=3")
print(f"  Language tags: <am>=4, <or>=5, <en>=6")
print(f"  SentencePiece tokens: 7-{VOCAB_SIZE-1}")

# ============ ENCODE/DECODE HELPERS (same as original) ============

def encode_sentence(text, lang, max_len=CONFIG['max_seq_length']):
    """Encode text with language-specific tokenizer, return unified IDs."""
    pieces = tokenizer.encode(str(text), out_type=str)
    ids = [TOKEN2ID.get(p, UNK_ID) for p in pieces]
    ids = ids[:max_len - 2]
    return [BOS_ID] + ids + [EOS_ID]

def decode_sentence(ids, lang):
    """Decode unified IDs back to text."""
    pieces = []
    for i in ids:
        if i in (PAD_ID, BOS_ID, EOS_ID):
            continue
        token = ID2TOKEN.get(i, "<unk>")
        if token in LANG_TAGS.values():
            continue
        pieces.append(token)
    return tokenizer.decode(pieces)

# ============ DATA LOADING ============
print("\n[2/7] Loading training data...")

aug_df = pd.read_csv(CONFIG['augmented_data_path'])
en_am_df = pd.read_csv(CONFIG['en_am_path'])
am_or_df = pd.read_csv(CONFIG['am_or_path'])

print(f"✓ Augmented trilingual: {len(aug_df)} rows")
print(f"✓ EN-AM bilingual: {len(en_am_df)} rows")
print(f"✓ AM-OR bilingual: {len(am_or_df)} rows")

# ============ DATASET WITH LANGUAGE TAGS ============
class MixedTrilingualDataset(Dataset):
    def __init__(self, aug_data, en_am_data, am_or_data, max_len=50):
        self.aug_data = aug_data.reset_index(drop=True)
        self.en_am_data = en_am_data.reset_index(drop=True)
        self.am_or_data = am_or_data.reset_index(drop=True)
        self.max_len = max_len
        
        print("  Pre-tokenizing datasets...")
        
        # Trilingual: all 6 directions
        self.tri_samples = []
        for _, row in self.aug_data.iterrows():
            en, am, or_ = row['English'], row['Amharic'], row['Oromo']
            # 6 directions
            self.tri_samples.append(('en', 'am', en, am))
            self.tri_samples.append(('am', 'en', am, en))
            self.tri_samples.append(('am', 'or', am, or_))
            self.tri_samples.append(('or', 'am', or_, am))
            self.tri_samples.append(('en', 'or', en, or_))
            self.tri_samples.append(('or', 'en', or_, en))
        
        # Bilingual
        self.enam_samples = []
        for _, row in self.en_am_data.iterrows():
            self.enam_samples.append(('en', 'am', row['English'], row['Amharic']))
            self.enam_samples.append(('am', 'en', row['Amharic'], row['English']))
        
        self.amor_samples = []
        for _, row in self.am_or_data.iterrows():
            self.amor_samples.append(('am', 'or', row[self.am_or_data.columns[0]], row[self.am_or_data.columns[1]]))
            self.amor_samples.append(('or', 'am', row[self.am_or_data.columns[1]], row[self.am_or_data.columns[0]]))
        
        print(f"  ✓ Trilingual pairs: {len(self.tri_samples)}")
        print(f"  ✓ EN-AM pairs: {len(self.enam_samples)}")
        print(f"  ✓ AM-OR pairs: {len(self.amor_samples)}")
    
    def __len__(self):
        return CONFIG['epoch_size']
    
    def __getitem__(self, idx):
        rand = np.random.random()
        
        if rand < 0.40:  # Trilingual (40%)
            sample = self.tri_samples[np.random.randint(0, len(self.tri_samples))]
            pair_type = 'trilingual'
            weight = 1.5
            
        elif rand < 0.70:  # EN-AM (30%)
            sample = self.enam_samples[np.random.randint(0, len(self.enam_samples))]
            pair_type = 'en_am'
            weight = 1.0
            
        else:  # AM-OR (30%)
            sample = self.amor_samples[np.random.randint(0, len(self.amor_samples))]
            pair_type = 'am_or'
            weight = 1.0
        
        src_lang, tgt_lang, src_text, tgt_text = sample
        
        # Encode source
        src_ids = encode_sentence(src_text, src_lang, self.max_len)
        
        # Encode target WITH LANGUAGE TAG
        tgt_ids = encode_sentence(tgt_text, tgt_lang, self.max_len)
        tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
        tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:]  # Replace duplicate BOS with tag
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'pair_type': pair_type,
            'loss_weight': weight,
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
        }

dataset = MixedTrilingualDataset(aug_df, en_am_df, am_or_df)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=2, pin_memory=False)

print(f"✓ Dataset: {len(dataset)} samples/epoch")

# ============ VERIFY BATCH COMPOSITION ============
print("\n[3/7] Verifying batch composition...")
sample_types = []
for i, batch in enumerate(dataloader):
    if i >= 10:
        break
    types = batch['pair_type']
    if isinstance(types, torch.Tensor):
        types = types.tolist()
    sample_types.extend(types)

counts = Counter(sample_types)
total = sum(counts.values())
for pt, c in counts.items():
    print(f"  {pt}: {c}/{total} ({c/total*100:.1f}%)")

# ============ MODEL (with correct vocab size) ============
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
        return [p for n, p in self.named_parameters() if 'encoder.' not in n]

# ============ LOAD BASELINE MODEL ============
print("\n[4/7] Loading baseline model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTransformer(VOCAB_SIZE, CONFIG['d_model'], CONFIG['n_heads'], CONFIG['n_layers'])

# Try to load baseline (may fail if vocab sizes differ, that's OK)
if os.path.exists(CONFIG['model_path']):
    checkpoint = torch.load(CONFIG['model_path'], map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state'])
                print("✓ Loaded baseline model")
            except Exception as e:
                print(f"⚠ Could not load baseline: {e}")
                print("  Training from scratch with language tags")
        else:
            print("⚠ Unknown checkpoint format, training from scratch")
    else:
        print("⚠ Invalid checkpoint, training from scratch")
else:
    print("⚠ No baseline found, training from scratch")

model = model.to(device)

# ============ TRAINING WITH FREEZING ============
print("\n[5/7] Training with periodic encoder freezing...")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
losses = []
freeze_log = []
best_loss = float('inf')

start_time = time.time()

for epoch in range(CONFIG['epochs']):
    epoch_start = time.time()
    
    should_freeze = (epoch % CONFIG['freeze_every_n_epochs'] == 0)
    
    if should_freeze:
        print(f"\n  >>> EPOCH {epoch+16}: ENCODER FROZEN <<<")
        optimizer = optim.Adam(model.get_decoder_params(), lr=CONFIG['learning_rate'])
        freeze_log.append(f"Epoch {epoch+16}: FROZEN")
    else:
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
        freeze_log.append(f"Epoch {epoch+16}: UNFROZEN")
    
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
        
        if (batch_idx + 1) % 500 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    
    # Time tracking
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    avg_epoch_time = sum([epoch_time]) / (epoch + 1)
    remaining = avg_epoch_time * (CONFIG['epochs'] - epoch - 1)
    
    def fmt(t):
        return f"{int(t//3600)}h {int((t%3600)//60)}m"
    
    status = " [FROZEN]" if should_freeze else ""
    print(f"  Epoch {epoch+16:2d}: Loss = {avg_loss:.4f} | Time: {fmt(epoch_time)} | Elapsed: {fmt(elapsed)} | ETA: {fmt(remaining)}{status}")
    
    # Save best
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save({
            'epoch': epoch + 16,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'config': CONFIG,
            'token2id': TOKEN2ID,
            'id2token': ID2TOKEN,
            'vocab_size': VOCAB_SIZE,
        }, f"{CONFIG['output_dir']}/trilingual_translator_v1_best.pt")
        print(f"    ✓ New best saved (loss: {avg_loss:.4f})")

# Final save
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'epoch': 16 + CONFIG['epochs'],
    'final_loss': losses[-1],
    'token2id': TOKEN2ID,
    'id2token': ID2TOKEN,
    'vocab_size': VOCAB_SIZE,
}, f"{CONFIG['output_dir']}/trilingual_translator_v1.pt")

# ============ PLOT & REPORT ============
print("\n[6/7] Generating plots...")

plt.figure(figsize=(12, 6))
plt.plot(range(16, 16+CONFIG['epochs']), losses, marker='o', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('Milestone 2.2: Training with Language Tags & Encoder Freezing', fontsize=14)
plt.grid(True, alpha=0.3)

for i, log in enumerate(freeze_log):
    if 'FROZEN' in log:
        plt.axvline(x=16+i, color='red', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/training_curves_trilingual.png", dpi=300)

report = {
    "milestone": "2.2 - Encoder Freezing (WITH LANGUAGE TAGS)",
    "freeze_schedule": freeze_log,
    "losses": [float(l) for l in losses],
    "vocab_size": VOCAB_SIZE,
    "config": CONFIG,
}

with open(f"{CONFIG['output_dir']}/milestone_2_2_report.json", 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "="*80)
print("MILESTONE 2.2 COMPLETE")
print("="*80)
print(f"✓ Best model: {CONFIG['output_dir']}/trilingual_translator_v1_best.pt")
print(f"✓ Final model: {CONFIG['output_dir']}/trilingual_translator_v1.pt")
print(f"✓ Vocab size: {VOCAB_SIZE} (with language tags)")
print("="*80)