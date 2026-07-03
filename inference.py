#!/usr/bin/env python3
"""
Phase 5 - Milestone 3.1: Inference Script
Standalone inference for the ALiBi multilingual translator.
Supports greedy and beam search decoding for EN↔AM and AM↔OR.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import sentencepiece as spm
from collections import defaultdict

# ============ CONFIG (will be overridden by checkpoint) ============
CONFIG = {
    'd_model': 256,
    'n_heads': 8,
    'n_layers': 4,
    'dim_feedforward': 1024,
    'dropout': 0.15,
    'max_seq_length': 128,
}

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

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
        
        slopes = [2 ** (-(8 / nhead) * (i + 1)) for i in range(nhead)]
        self.register_buffer('slopes', torch.tensor(slopes, dtype=torch.float32))
    
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        batch_size, q_len, _ = query.size()
        k_len = key.size(1)
        
        Q = self.q_proj(query).view(batch_size, q_len, self.nhead, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, k_len, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        q_pos = torch.arange(q_len, device=scores.device).unsqueeze(1).float()
        k_pos = torch.arange(k_len, device=scores.device).unsqueeze(0).float()
        dist = torch.abs(q_pos - k_pos)
        alibi_bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0)
        scores = scores + alibi_bias.unsqueeze(0)
        
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
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_out))
        
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

# ============ TOKENIZATION ============
def encode_sentence(text, tokenizer, token2id, max_len=128):
    pieces = tokenizer.encode(str(text), out_type=str)
    ids = [token2id.get(p, UNK_ID) for p in pieces]
    ids = ids[:max_len - 2]
    ids = [BOS_ID] + ids + [EOS_ID]
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids[:max_len]

def decode_ids(ids, id2token, tokenizer):
    """Convert token IDs back to text, handling SentencePiece pieces."""
    pieces = []
    for tid in ids:
        if tid in (PAD_ID, BOS_ID, EOS_ID, UNK_ID):
            continue
        token = id2token.get(tid, "")
        if token.startswith("<") and token.endswith(">"):
            continue  # Skip language tags
        pieces.append(token)
    return tokenizer.decode(pieces)

# ============ DECODING ============
def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
    return mask.bool()

@torch.no_grad()
def greedy_decode(model, src_tensor, tgt_lang, token2id, id2token, tokenizer,
                  max_len=128, ngram_block=3, device='cpu'):
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)
    src_pad_mask = (src == PAD_ID)
    
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    tgt_tag_id = token2id[LANG_TAGS[tgt_lang]]
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
def beam_search_decode(model, src_tensor, tgt_lang, token2id, id2token, tokenizer,
                       beam_width=5, max_len=128, ngram_block=3,
                       repetition_penalty=1.2, length_penalty=0.8, device='cpu'):
    model.eval()
    src = src_tensor.unsqueeze(0).to(device)
    src_pad_mask = (src == PAD_ID)
    
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)
    tgt_tag_id = token2id[LANG_TAGS[tgt_lang]]
    
    beams = [([BOS_ID, tgt_tag_id], 0.0, False)]
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

# ============ TRANSLATION INTERFACE ============
class MultilingualTranslator:
    """High-level interface for the ALiBi multilingual translator."""
    
    SUPPORTED_PAIRS = {
        ('en', 'am'): 'English → Amharic',
        ('am', 'en'): 'Amharic → English',
        ('am', 'or'): 'Amharic → Oromo',
        ('or', 'am'): 'Oromo → Amharic',
    }
    
    def __init__(self, checkpoint_path, tokenizer_path='output/spm_unified_multilingual.model',
                 device=None, beam_width=5, max_len=128):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.beam_width = beam_width
        self.max_len = max_len
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.token2id = checkpoint['token2id']
        self.id2token = checkpoint['id2token']
        self.vocab_size = checkpoint['vocab_size']
        
        # Override config with checkpoint config if available
        cfg = checkpoint.get('config', CONFIG)
        self.d_model = cfg.get('d_model', 256)
        self.n_heads = cfg.get('n_heads', 8)
        self.n_layers = cfg.get('n_layers', 4)
        self.dim_feedforward = cfg.get('dim_feedforward', 1024)
        self.dropout = cfg.get('dropout', 0.15)
        self.max_seq_length = cfg.get('max_seq_length', 128)
        
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Architecture: d_model={self.d_model}, heads={self.n_heads}, layers={self.n_layers}")
        print(f"  Device: {self.device}")
        
        self.model = ALiBiTransformer(
            self.vocab_size, self.d_model, self.n_heads, self.n_layers,
            self.dim_feedforward, self.dropout, self.max_seq_length
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print("  ✓ Model loaded successfully")
        
        print(f"Loading tokenizer from: {tokenizer_path}")
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        print("  ✓ Tokenizer loaded successfully")
    
    def translate(self, text, src_lang, tgt_lang, method='beam', **kwargs):
        """
        Translate text from src_lang to tgt_lang.
        
        Args:
            text: Source text string
            src_lang: Source language code ('en', 'am', 'or')
            tgt_lang: Target language code ('en', 'am', 'or')
            method: 'beam' or 'greedy'
            **kwargs: Override beam_width, max_len, repetition_penalty, length_penalty
        
        Returns:
            Translated text string
        """
        src_lang = src_lang.lower()
        tgt_lang = tgt_lang.lower()
        
        if (src_lang, tgt_lang) not in self.SUPPORTED_PAIRS:
            supported = [f"{s}→{t}" for s, t in self.SUPPORTED_PAIRS.keys()]
            raise ValueError(f"Unsupported pair: {src_lang}→{tgt_lang}. Supported: {', '.join(supported)}")
        
        src_ids = encode_sentence(text, self.tokenizer, self.token2id, self.max_seq_length)
        src_tensor = torch.tensor(src_ids, dtype=torch.long)
        
        beam_width = kwargs.get('beam_width', self.beam_width)
        max_len = kwargs.get('max_len', self.max_len)
        rep_penalty = kwargs.get('repetition_penalty', 1.2)
        len_penalty = kwargs.get('length_penalty', 0.8)
        
        if method == 'beam':
            hyp_ids = beam_search_decode(
                self.model, src_tensor, tgt_lang, self.token2id, self.id2token, self.tokenizer,
                beam_width=beam_width, max_len=max_len,
                repetition_penalty=rep_penalty, length_penalty=len_penalty,
                device=self.device
            )
        else:
            hyp_ids = greedy_decode(
                self.model, src_tensor, tgt_lang, self.token2id, self.id2token, self.tokenizer,
                max_len=max_len, device=self.device
            )
        
        return decode_ids(hyp_ids, self.id2token, self.tokenizer)
    
    def translate_batch(self, texts, src_lang, tgt_lang, method='beam', **kwargs):
        """Translate a batch of texts."""
        return [self.translate(t, src_lang, tgt_lang, method, **kwargs) for t in texts]
    
    def interactive(self):
        """Run interactive translation shell."""
        print("\n" + "="*60)
        print("MULTILINGUAL TRANSLATOR - INTERACTIVE MODE")
        print("="*60)
        print("Supported directions: EN↔AM, AM↔OR")
        print("Commands: 'quit' to exit, 'beam=N' to set beam width")
        print("-"*60)
        
        current_beam = self.beam_width
        
        while True:
            try:
                user_input = input("\n[en/am/or] src→tgt: ").strip()
                
                if user_input.lower() in ('quit', 'exit', 'q'):
                    break
                
                if user_input.lower().startswith('beam='):
                    try:
                        current_beam = int(user_input.split('=')[1])
                        print(f"  Beam width set to {current_beam}")
                    except:
                        print("  Invalid beam width")
                    continue
                
                # Parse direction: e.g., "en→am" or "en->am" or "en am"
                direction = user_input.replace('->', '→').replace(' ', '→')
                if '→' not in direction:
                    print("  Format: src→tgt (e.g., en→am)")
                    continue
                
                src_lang, tgt_lang = direction.split('→', 1)
                src_lang = src_lang.strip().lower()
                tgt_lang = tgt_lang.strip().lower()
                
                text = input("Text: ").strip()
                if not text:
                    continue
                
                print(f"  Translating ({'beam' if current_beam > 1 else 'greedy'})...")
                result = self.translate(text, src_lang, tgt_lang, 
                                        method='beam' if current_beam > 1 else 'greedy',
                                        beam_width=current_beam)
                print(f"  Result: {result}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"  Error: {e}")

# ============ CLI ============
def main():
    parser = argparse.ArgumentParser(description='Multilingual Translator Inference')
    parser.add_argument('--checkpoint', '-c', required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--tokenizer', '-t', default='output/spm_unified_multilingual.model',
                        help='Path to SentencePiece model')
    parser.add_argument('--text', help='Text to translate (if not provided, enters interactive mode)')
    parser.add_argument('--src', '-s', choices=['en', 'am', 'or'], help='Source language')
    parser.add_argument('--tgt', '-T', choices=['en', 'am', 'or'], help='Target language')
    parser.add_argument('--method', '-m', choices=['greedy', 'beam'], default='beam',
                        help='Decoding method')
    parser.add_argument('--beam-width', '-b', type=int, default=5, help='Beam width')
    parser.add_argument('--max-len', type=int, default=128, help='Max sequence length')
    parser.add_argument('--repetition-penalty', '-r', type=float, default=1.2)
    parser.add_argument('--length-penalty', '-l', type=float, default=0.8)
    parser.add_argument('--device', '-d', default=None, help='Device (cuda/cpu)')
    parser.add_argument('--interactive', '-i', action='store_true', help='Force interactive mode')
    
    args = parser.parse_args()
    
    translator = MultilingualTranslator(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=args.device,
        beam_width=args.beam_width,
        max_len=args.max_len
    )
    
    if args.interactive or (args.text is None):
        translator.interactive()
    else:
        if not args.src or not args.tgt:
            parser.error("--src and --tgt required when --text is provided")
        
        result = translator.translate(
            args.text, args.src, args.tgt, method=args.method,
            beam_width=args.beam_width, max_len=args.max_len,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty
        )
        print(result)

if __name__ == '__main__':
    main()