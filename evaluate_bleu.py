#!/usr/bin/env python3
"""
BLEU Evaluation Script for Phase 5 Final Model

Evaluates all 6 translation directions using sacrebleu.
Requires: pip install sacrebleu

Usage:
    python evaluate_bleu.py \
        --model output/phase_5_milestone_3_1/final_translator_best.pt \
        --test-data output/data_final/clean_translation_final.csv \
        --output bleu_results.json
"""

import argparse
import json
import torch
import sentencepiece as spm
import sacrebleu
from typing import List, Dict


# ============ MODEL (same as inference_final.py) ============

import torch.nn as nn

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_LEN = 80
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        enc_out = self.encoder(src_emb)
        dec_out = self.decoder(tgt_emb, enc_out)
        return self.output_projection(dec_out)


# ============ TRANSLATOR ============

class Translator:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        self.vocab_size = self.sp.GetPieceSize()
        
        # Build vocab
        self.token2id = {self.sp.IdToPiece(i): i for i in range(self.vocab_size)}
        self.id2token = {i: self.sp.IdToPiece(i) for i in range(self.vocab_size)}
        
        # Load model
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            config = ckpt.get('config', {})
        elif isinstance(ckpt, dict) and 'model_state' in ckpt:
            state = ckpt['model_state']
            config = ckpt.get('config', {})
        else:
            state = ckpt
            config = {}
        
        self.model = SimpleTransformer(
            self.vocab_size,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 2),
        ).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def encode(self, text: str) -> List[int]:
        pieces = self.sp.encode(str(text), out_type=str)
        ids = [self.token2id.get(p, UNK_ID) for p in pieces]
        ids = ids[:MAX_LEN - 2]
        return [BOS_ID] + ids + [EOS_ID]

    def decode(self, ids: List[int]) -> str:
        valid = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            piece = self.id2token.get(i, "<unk>")
            if piece in ('<am>', '<or>', '<en>'):
                continue
            valid.append(i)
        return self.sp.decode(valid)

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        src_ids = self.encode(text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        
        tgt_tag_id = self.token2id.get(LANG_TAGS[tgt_lang], UNK_ID)
        tgt_ids = [BOS_ID, tgt_tag_id]
        
        with torch.no_grad():
            for _ in range(MAX_LEN):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
                logits = self.model(src_tensor, tgt_tensor)
                next_id = logits[0, -1].argmax().item()
                if next_id == EOS_ID:
            # 
                  tgt_ids.append(next_id)
        
        return self.decode(tgt_ids)


# ============ EVALUATION ============

def evaluate_direction(translator: Translator, sources: List[str], references: List[str], 
                       src_lang: str, tgt_lang: str) -> Dict:
    """Evaluate one translation direction."""
    print(f"\n  Evaluating {src_lang} → {tgt_lang} ({len(sources)} sentences)...")
    
    hypotheses = []
    for i, src in enumerate(sources):
        hyp = translator.translate(src, src_lang, tgt_lang)
        hypotheses.append(hyp)
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(sources)} done")
    
    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    
    # Compute chrF
    chrf = sacrebleu.corpus_chrf(hypotheses, [references])
    
    # Show 3 examples
    print(f"\n    Examples:")
    for s, h, r in zip(sources[:3], hypotheses[:3], references[:3]):
        print(f"      SRC: {s[:60]}")
        print(f"      HYP: {h[:60]}")
        print(f"      REF: {r[:60]}")
        print()
    
    return {
        "bleu": bleu.score,
        "bleu_signature": bleu.signature,
        "chrf": chrf.score,
        "chrf_signature": chrf.signature,
        "num_sentences": len(sources),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="output/spm_unified_multilingual.model")
    parser.add_argument("--test-data", required=True, help="Path to test CSV (trilingual)")
    parser.add_argument("--output", default="bleu_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("BLEU EVALUATION — Phase 5 Final Model")
    print("=" * 60)

    # Load model
    print(f"\n[1/3] Loading model from {args.model}...")
    translator = Translator(args.model, args.tokenizer)

    # Load test data
    print(f"\n[2/3] Loading test data from {args.test_data}...")
    import pandas as pd
    df = pd.read_csv(args.test_data)
    print(f"  Loaded {len(df)} sentences")
    print(f"  Columns: {list(df.columns)}")

    # Auto-detect columns
    cols = list(df.columns)
    col_map = {}
    col_map = {'en': 'English', 'am': 'Amharic', 'or': 'Oromo'}
    pass  # Hardcoded columns
        # Auto-detection replaced with hardcoded
        # See col_map above
        # 
    
    print(f"  Detected: {col_map}")

    # Use last 20% as test set (hold-out)
    test_size = int(len(df) * 0.2)
    test_df = df.tail(test_size).reset_index(drop=True)
    print(f"  Using {len(test_df)} sentences for evaluation (last 20%)")

    # Evaluate all 6 directions
    print("\n[3/3] Evaluating all 6 translation directions...")
    
    results = {}
    
    directions = [
        ("en", "am", col_map['en'], col_map['am']),
        ("am", "en", col_map['am'], col_map['en']),
        ("am", "or", col_map['am'], col_map['or']),
        ("or", "am", col_map['or'], col_map['am']),
        ("en", "or", col_map['en'], col_map['or']),
        ("or", "en", col_map['or'], col_map['en']),
    ]
    
    for src_lang, tgt_lang, src_col, tgt_col in directions:
        sources = test_df[src_col].astype(str).tolist()
        references = test_df[tgt_col].astype(str).tolist()
        
        results[f"{src_lang}_{tgt_lang}"] = evaluate_direction(
            translator, sources, references, src_lang, tgt_lang
        )

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    for direction, metrics in results.items():
        print(f"\n  {direction}:")
        print(f"    BLEU:  {metrics['bleu']:.2f}")
        print(f"    chrF:  {metrics['chrf']:.2f}")
        print(f"    Sents: {metrics['num_sentences']}")

    # Save
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {args.output}")

    # Average BLEU
    avg_bleu = sum(m['bleu'] for m in results.values()) / len(results)
    print(f"\n📊 Average BLEU across 6 directions: {avg_bleu:.2f}")


if __name__ == "__main__":
    main()