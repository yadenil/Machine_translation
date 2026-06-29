#!/usr/bin/env python3
"""
Phase 5 - Milestone 3.1: Convergence (UPGRADED)
Changes:
  - 4 directions only (EN↔AM, AM↔OR) — EN↔OR removed
  - ALiBi positional encoding (custom self-attention)
  - Scaled architecture: d_model=256, nhead=8, n_layers=4, dim_ff=1024, dropout=0.15
  - Pre-split data from phase_5_milestone_3_1/ with 60/40 EN-AM/AM-OR sampling
  - AdamW + 2-epoch warmup + step decay, label smoothing 0.1, grad clipping 1.0
  - Beam search (width=5) for final evaluation; greedy kept for fast val BLEU
  - Fresh training run — old checkpoint incompatible
"""

# ============ TENSORBOARD USAGE ============
#   tensorboard --logdir=output/phase_5_milestone_3_1/tensorboard
# ===========================================

import os
import json
import time
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict, Counter
from torch.utils.tensorboard import SummaryWriter

CONFIG = {
    'output_dir': 'output/phase_5_milestone_3_1',
    'start_epoch': 1,
    'epochs': 20,
    'batch_size': 64,
    'base_lr': 3e-4,
    'lr_warmup_epochs': 2,
    'lr_decay_factor': 0.1,
    'lr_decay_every': 5,
    'epoch_size': 80000,
    'en_am_pct': 0.60,
    'am_or_pct': 0.40,
    'patience': 4,
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dim_feedforward': 1024,
    'dropout': 0.15,
    'max_seq_length': 128,
    'tokenizer_path': 'output/spm_unified_multilingual.model',
    'val_bleu_interval': 2,
    'val_bleu_samples': 200,
    'label_smoothing': 0.1,
    'grad_clip_norm': 1.0,
    'beam_width': 5,
    'repetition_penalty': 1.2,
    'length_penalty': 0.8,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

tb_log_dir = os.path.join(CONFIG['output_dir'], 'tensorboard')
writer = SummaryWriter(log_dir=tb_log_dir)
print(f"📊 TensorBoard log dir: {tb_log_dir}")
print(f"   Launch with: tensorboard --logdir={tb_log_dir}")

import sentencepiece as spm

print("="*80)
print("PHASE 5 - MILESTONE 3.1: CONVERGENCE (UPGRADED — 4 DIRECTIONS, ALiBi, ADAMW)")
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
    ids = ids[:max_len - 2]
    ids = [BOS_ID] + ids + [EOS_ID]
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids[:max_len]

# ============ DATA LOADING (PRE-SPLIT) ============
print("\n[2/8] Loading pre-split datasets...")

split_dir = CONFIG['output_dir']

tri_train = pd.read_csv(f'{split_dir}/train_trilingual.csv')
en_am_train = pd.read_csv(f'{split_dir}/train_en_am.csv')
am_or_train = pd.read_csv(f'{split_dir}/train_am_or.csv')

tri_val = pd.read_csv(f'{split_dir}/val_trilingual.csv')
en_am_val = pd.read_csv(f'{split_dir}/val_en_am.csv')
am_or_val = pd.read_csv(f'{split_dir}/val_am_or.csv')

print(f"\n  Loaded splits:")
print(f"    Trilingual: train={len(tri_train)}, val={len(tri_val)}")
print(f"    EN-AM:      train={len(en_am_train)}, val={len(en_am_val)}")
print(f"    AM-OR:      train={len(am_or_train)}, val={len(am_or_val)}")

# ============ DATASET ============
class ConvergenceDataset(Dataset):
    def __init__(self, en_am_data, am_or_data, tri_data=None):
        self.en_am_samples = []
        self.am_or_samples = []
        
        print("  Building training pools...")
        
        # EN-AM bilingual pool
        for _, row in en_am_data.iterrows():
            self.en_am_samples.append(('en', 'am', row['English'], row['Amharic']))
            self.en_am_samples.append(('am', 'en', row['Amharic'], row['English']))
        
        # AM-OR bilingual pool
        for _, row in am_or_data.iterrows():
            c = list(row)
            self.am_or_samples.append(('am', 'or', c[0], c[1]))
            self.am_or_samples.append(('or', 'am', c[1], c[0]))
        
        # Decompose trilingual into EN-AM and AM-OR only (no EN-OR)
        if tri_data is not None:
            for _, row in tri_data.iterrows():
                en, am, or_ = row['English'], row['Amharic'], row['Oromo']
                self.en_am_samples.append(('en', 'am', en, am))
                self.en_am_samples.append(('am', 'en', am, en))
                self.am_or_samples.append(('am', 'or', am, or_))
                self.am_or_samples.append(('or', 'am', or_, am))
        
        print(f"  ✓ EN-AM pool: {len(self.en_am_samples)} directionals")
        print(f"  ✓ AM-OR pool: {len(self.am_or_samples)} directionals")
    
    def __len__(self):
        return CONFIG['epoch_size']
    
    def __getitem__(self, idx):
        r = np.random.random()
        
        # 60% EN-AM / 40% AM-OR
        if r < CONFIG['en_am_pct']:
            s = self.en_am_samples[np.random.randint(0, len(self.en_am_samples))]
            pair_type = 'en_am'
        else:
            s = self.am_or_samples[np.random.randint(0, len(self.am_or_samples))]
            pair_type = 'am_or'
        
        src_lang, tgt_lang, src_text, tgt_text = s
        
        src_ids = encode_sentence(src_text, src_lang)
        tgt_ids = encode_sentence(tgt_text, tgt_lang)
        tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
        
        # Prepend language tag after BOS
        tgt_ids = [BOS_ID, tgt_tag_id] + tgt_ids[1:-1] + [EOS_ID]
        while len(tgt_ids) < CONFIG['max_seq_length']:
            tgt_ids.append(PAD_ID)
        tgt_ids = tgt_ids[:CONFIG['max_seq_length']]
        
        return {
            'src': torch.tensor(src_ids, dtype=torch.long),
            'tgt': torch.tensor(tgt_ids, dtype=torch.long),
            'pair_type': pair_type,
        }

dataset = ConvergenceDataset(en_am_train, am_or_train, tri_data=tri_train)
dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True,
                        num_workers=2, pin_memory=False)

print(f"✓ Dataset: {len(dataset)} samples/epoch")

# ============ ALiBi MULTI-HEAD ATTENTION ============
class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # ALiBi slopes: geometric sequence 2^(-(8/n)*(i+1))
        slopes = [2 ** (-(8 / nhead) * (i + 1)) for i in range(nhead)]
        self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        
        Q = self.q_proj(query).view(batch_size, q_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, Q, K)
        
        # ALiBi bias: per-head linear distance penalty
        q_pos = torch.arange(q_len, device=scores.device).unsqueeze(1).float()
        k_pos = torch.arange(k_len, device=scores.device).unsqueeze(0).float()
        dist = torch.abs(q_pos - k_pos)  # (Q, K)
        alibi_bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)  # (H, Q, K)
        scores = scores + alibi_bias.unsqueeze(0)  # (B, H, Q, K)
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            else:
                scores = scores.masked_fill(attn_mask, float('-inf'))
        
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.out_proj(out)
        return out

# ============ CUSTOM TRANSFORMER BLOCKS ============
class ALiBiEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.15):
        super().__init__()
        self.self_attn = ALiBiMultiHeadAttention(d_model, nhead, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_out = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = self.norm1(src + self.dropout1(attn_out))
        ff_out = self.ff(src)
        src = self.norm2(src + self.dropout2(ff_out))
        return src

class ALiBiDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.15):
        super().__init__()
        self.self_attn = ALiBiMultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Self-attention with ALiBi
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
        # Cross-attention (standard)
        cross_out, _ = self.cross_attn(tgt, memory, memory,
                                       attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask,
                                       need_weights=False)
        tgt = self.norm2(tgt + self.dropout2(cross_out))
        
        ff_out = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout3(ff_out))
        return tgt

# ============ MODEL ============
class ALiBiTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, n_layers=4,
                 dim_feedforward=1024, dropout=0.15, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.encoder_layers = nn.ModuleList([
            ALiBiEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
            ALiBiDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, src, src_key_padding_mask=None):
        src_emb = self.dropout(self.embedding(src))
        memory = src_emb
        for layer in self.encoder_layers:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode_step(self, tgt, memory, tgt_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_emb = self.dropout(self.embedding(tgt))
        out = tgt_emb
        for layer in self.decoder_layers:
            out = layer(out, memory,
                       tgt_mask=tgt_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask,
                       memory_key_padding_mask=memory_key_padding_mask)
        return self.output_projection(out)
    
    def forward(self, src, tgt, tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)
        return self.decode_step(tgt, memory,
                                tgt_mask=tgt_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ============ LOAD CHECKPOINT (SKIPPED — FRESH RUN) ============
print("\n[3/8] Initializing fresh model (architecture changed — old checkpoint incompatible)...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ALiBiTransformer(VOCAB_SIZE, CONFIG['d_model'], CONFIG['n_heads'],
                         CONFIG['n_layers'], CONFIG['dim_feedforward'],
                         CONFIG['dropout'], CONFIG['max_seq_length'])

print(f"  ✓ Model initialized: {count_parameters(model):,} parameters (~{count_parameters(model)/1e6:.1f}M)")
print(f"  ⚠ Old checkpoint NOT loaded — fresh training from epoch {CONFIG['start_epoch']}")

model = model.to(device)

# ============ TRAINING ============
print("\n[4/8] Starting convergence training...")

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=CONFIG['label_smoothing'])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['base_lr'],
                          betas=(0.9, 0.98), weight_decay=0.01)

losses = []
val_bleus = []
lr_history = []
best_epoch = 0
best_loss = float('inf')
patience_counter = 0

start_time = time.time()

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()

# ============ DECODING ============
@torch.no_grad()
def greedy_decode(model, src_tensor, tgt_lang, max_len=128, ngram_block=3):
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)
    src_pad_mask = (src == PAD_ID)
    
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
    generated = [BOS_ID, tgt_tag_id]
    seen_ngrams = set()
    
    for _ in range(max_len - 2):
        tgt_tensor = torch.tensor([generated], dtype=torch.long, device=device)
        tgt_len = tgt_tensor.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        tgt_pad_mask = (tgt_tensor == PAD_ID)
        
        logits = model.decode_step(tgt_tensor, memory,
                                   tgt_mask=tgt_mask,
                                   tgt_key_padding_mask=tgt_pad_mask,
                                   memory_key_padding_mask=src_pad_mask)
        
        next_token_logits = logits[0, -1, :].clone()
        
        if len(generated) >= ngram_block - 1:
            prefix = tuple(generated[-(ngram_block - 1):])
            for token_id in range(next_token_logits.size(0)):
                candidate_ngram = prefix + (token_id,)
                if candidate_ngram in seen_ngrams:
                    next_token_logits[token_id] = float('-inf')
        
        next_token = next_token_logits.argmax().item()
        
        if next_token == EOS_ID:
            break
        generated.append(next_token)
        
        if len(generated) >= ngram_block:
            seen_ngrams.add(tuple(generated[-ngram_block:]))
    
    return generated[2:]

@torch.no_grad()
def beam_search_decode(model, src_tensor, tgt_lang, beam_width=5, max_len=128,
                       ngram_block=3, repetition_penalty=1.2, length_penalty=0.8):
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)
    src_pad_mask = (src == PAD_ID)
    
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    tgt_tag_id = TOKEN2ID[LANG_TAGS[tgt_lang]]
    
    beams = [([BOS_ID, tgt_tag_id], 0.0, False)]  # (tokens, raw_score, done)
    completed = []
    
    for _ in range(max_len - 2):
        candidates = []
        for tokens, score, done in beams:
            if done:
                lp = ((5 + len(tokens)) / 6.0) ** length_penalty
                candidates.append((tokens, score, True, score / lp))
                continue
            
            tgt_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
            tgt_len = len(tokens)
            tgt_mask = generate_square_subsequent_mask(tgt_len, device)
            tgt_pad_mask = (tgt_tensor == PAD_ID)
            
            logits = model.decode_step(tgt_tensor, memory,
                                       tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_pad_mask,
                                       memory_key_padding_mask=src_pad_mask)
            
            next_token_logits = logits[0, -1, :].clone()
            
            for token in set(tokens):
                next_token_logits[token] /= repetition_penalty
            
            if len(tokens) >= ngram_block - 1:
                prefix = tuple(tokens[-(ngram_block - 1):])
                for token_id in range(next_token_logits.size(0)):
                    candidate_ngram = prefix + (token_id,)
                    exists = False
                    for i in range(len(tokens) - ngram_block + 1):
                        if tuple(tokens[i:i+ngram_block]) == candidate_ngram:
                            exists = True
                            break
                    if exists:
                        next_token_logits[token_id] = float('-inf')
            
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_width * 2)
            
            for log_prob, token_id in zip(topk_log_probs, topk_ids):
                new_tokens = tokens + [token_id.item()]
                new_score = score + log_prob.item()
                if token_id.item() == EOS_ID:
                    lp = ((5 + len(new_tokens)) / 6.0) ** length_penalty
                    candidates.append((new_tokens, new_score, True, new_score / lp))
                else:
                    candidates.append((new_tokens, new_score, False, new_score))
        
        candidates.sort(key=lambda x: x[3], reverse=True)
        top_candidates = candidates[:beam_width]
        
        beams = []
        for tokens, score, done, _ in top_candidates:
            if done:
                completed.append((tokens, score))
            else:
                beams.append((tokens, score, done))
        
        if not beams:
            break
    
    for tokens, score, done in beams:
        lp = ((5 + len(tokens)) / 6.0) ** length_penalty
        completed.append((tokens, score / lp))
    
    if not completed:
        return []
    
    completed.sort(key=lambda x: x[1], reverse=True)
    return completed[0][0][2:]

# ============ BLEU ============
def compute_bleu(references, hypotheses, max_n=4):
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
    
    precisions = []
    for n in range(max_n):
        if total_counts[n] == 0:
            return 0.0
        precisions.append(clipped_counts[n] / total_counts[n])
    
    log_avg = sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / max_n
    if log_avg == float('-inf'):
        return 0.0
    
    if hyp_len == 0:
        return 0.0
    bp = math.exp(min(0, 1 - ref_len / hyp_len))
    
    return bp * math.exp(log_avg)

@torch.no_grad()
def evaluate_bleu(model, tri_val, en_am_val, am_or_val, n_samples=CONFIG['val_bleu_samples'],
                  use_beam=False):
    model.eval()
    pair_results = defaultdict(lambda: {'refs': [], 'hyps': []})
    decode_fn = beam_search_decode if use_beam else greedy_decode
    
    # Trilingual validation (decomposed into 4 directions)
    tri_sample = tri_val.sample(n=min(n_samples // 4, len(tri_val)), random_state=None)
    for _, row in tri_sample.iterrows():
        en, am, or_ = row['English'], row['Amharic'], row['Oromo']
        for src_lang, tgt_lang, src_text, tgt_text in [
            ('en', 'am', en, am),
            ('am', 'en', am, en),
            ('am', 'or', am, or_),
            ('or', 'am', or_, am),
        ]:
            src_ids = encode_sentence(src_text, src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = decode_fn(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(tgt_text, tgt_lang)
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    # EN-AM validation
    enam_sample = en_am_val.sample(n=min(n_samples // 4, len(en_am_val)), random_state=None)
    for _, row in enam_sample.iterrows():
        for src_lang, tgt_lang, src_col, tgt_col in [
            ('en', 'am', 'English', 'Amharic'),
            ('am', 'en', 'Amharic', 'English'),
        ]:
            src_ids = encode_sentence(row[src_col], src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = decode_fn(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(row[tgt_col], tgt_lang)
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    # AM-OR validation
    amor_sample = am_or_val.sample(n=min(n_samples // 4, len(am_or_val)), random_state=None)
    for _, row in amor_sample.iterrows():
        c = list(row)
        for src_lang, tgt_lang, src_text, tgt_text in [
            ('am', 'or', c[0], c[1]),
            ('or', 'am', c[1], c[0]),
        ]:
            src_ids = encode_sentence(src_text, src_lang)
            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            hyp_ids = decode_fn(model, src_tensor, tgt_lang)
            ref_ids = encode_sentence(tgt_text, tgt_lang)
            ref_clean = [t for t in ref_ids if t not in (PAD_ID, BOS_ID, EOS_ID)]
            lang_tag_ids = set(TOKEN2ID[v] for v in LANG_TAGS.values())
            ref_clean = [t for t in ref_clean if t not in lang_tag_ids]
            pair_key = f"{src_lang.upper()}→{tgt_lang.upper()}"
            pair_results[pair_key]['refs'].append(ref_clean)
            pair_results[pair_key]['hyps'].append(hyp_ids)
    
    bleu_scores = {}
    for pair_key, data in sorted(pair_results.items()):
        bleu_scores[pair_key] = compute_bleu(data['refs'], data['hyps'])
    
    if bleu_scores:
        bleu_scores['AVG'] = sum(bleu_scores.values()) / len(bleu_scores)
    else:
        bleu_scores['AVG'] = 0.0
    
    model.train()
    return bleu_scores

# ============ TRAINING LOOP ============
for epoch_idx in range(CONFIG['epochs']):
    abs_epoch = CONFIG['start_epoch'] + epoch_idx
    
    # Warmup + step decay LR schedule
    if epoch_idx < CONFIG['lr_warmup_epochs']:
        current_lr = CONFIG['base_lr'] * (epoch_idx + 1) / CONFIG['lr_warmup_epochs']
    else:
        decay_steps = (epoch_idx - CONFIG['lr_warmup_epochs']) // CONFIG['lr_decay_every']
        current_lr = CONFIG['base_lr'] * (CONFIG['lr_decay_factor'] ** decay_steps)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    
    model.train()
    epoch_loss = 0
    batch_count = 0
    epoch_start = time.time()
    
    for batch_idx, batch in enumerate(dataloader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        
        optimizer.zero_grad()
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        tgt_len = tgt_input.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_len, device)
        src_key_padding_mask = (src == PAD_ID)
        tgt_key_padding_mask = (tgt_input == PAD_ID)
        
        logits = model(src, tgt_input,
                       tgt_mask=tgt_mask,
                       src_key_padding_mask=src_key_padding_mask,
                       tgt_key_padding_mask=tgt_key_padding_mask)
        
        logits = logits.reshape(-1, logits.size(-1))
        tgt_output = tgt_output.reshape(-1)
        
        loss = criterion(logits, tgt_output)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
        optimizer.step()
        
        epoch_loss += loss.item()
        batch_count += 1
        
        global_step = epoch_idx * len(dataloader) + batch_idx
        writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
        
        if (batch_idx + 1) % 500 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
            writer.add_scalar('Train/LR', current_lr, global_step)
    
    avg_loss = epoch_loss / batch_count
    losses.append(avg_loss)
    lr_history.append(current_lr)
    
    epoch_global_step = (epoch_idx + 1) * len(dataloader)
    writer.add_scalar('Train/Loss_epoch', avg_loss, epoch_global_step)
    writer.add_scalar('Train/LR', current_lr, epoch_global_step)
    
    # Validation BLEU (greedy — fast)
    val_bleu = 0.0
    bleu_str = ''
    if (epoch_idx + 1) % CONFIG['val_bleu_interval'] == 0 or epoch_idx == CONFIG['epochs'] - 1:
        print(f"    Evaluating greedy BLEU on validation set...")
        bleu_scores = evaluate_bleu(model, tri_val, en_am_val, am_or_val, use_beam=False)
        val_bleu = bleu_scores.get('AVG', 0.0)
        for pair_key, score in bleu_scores.items():
            tag = f"Val/BLEU_{pair_key.replace('→', '_to_')}"
            writer.add_scalar(tag, score, epoch_global_step)
        pair_strs = [f"{k}={v:.4f}" for k, v in bleu_scores.items()]
        print(f"    BLEU: {' | '.join(pair_strs)}")
        bleu_str = f" | BLEU={val_bleu:.4f}"
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
        writer.add_scalar('Train/Best_loss', avg_loss, epoch_global_step)
    else:
        patience_counter += 1
    
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - start_time
    
    def fmt(t):
        return f"{int(t//3600)}h {int((t%3600)//60)}m"
    
    print(f"  Epoch {abs_epoch:2d}: Loss={avg_loss:.4f} | LR={current_lr:.0e}{bleu_str}"
          f" | Time={fmt(epoch_time)} | Elapsed={fmt(elapsed)}")
    
    if patience_counter >= CONFIG['patience']:
        print(f"\n  Early stopping!")
        break

# ============ FINAL BEAM SEARCH EVALUATION ============
print("\n[5/8] Final beam search evaluation...")
beam_bleu = evaluate_bleu(model, tri_val, en_am_val, am_or_val,
                          n_samples=CONFIG['val_bleu_samples'], use_beam=True)
print("  Beam Search BLEU:")
for k, v in sorted(beam_bleu.items()):
    print(f"    {k}: {v:.4f}")
writer.add_scalar('Val/BLEU_beam_AVG', beam_bleu.get('AVG', 0.0), 0)

# ============ FINAL SAVE ============
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

writer.close()
print(f"\n📊 TensorBoard logs saved to: {tb_log_dir}")

print("\n" + "="*80)
print("MILESTONE 3.1 UPGRADE COMPLETE")
print("="*80)
print(f"✓ Final model: {CONFIG['output_dir']}/final_translator_multilingual.pt")
print(f"✓ Best model:  {CONFIG['output_dir']}/final_translator_best.pt")
print(f"✓ Vocab size: {VOCAB_SIZE} (with language tags)")
print(f"✓ Parameters:  {count_parameters(model):,} (~{count_parameters(model)/1e6:.1f}M)")
print(f"✓ Directions:  EN→AM, AM→EN, AM→OR, OR→AM (EN↔OR removed)")
print("="*80)