#!/usr/bin/env python3
"""
BLEU Evaluation Script — Phase 5 Final Model
Evaluates on HELD-OUT test set (80/10/10 split)

This provides HONEST BLEU scores on unseen data that was held out before training.

Usage:
    python evaluate_bleu.py \
        --model output/phase_5_milestone_3_1/final_translator_best.pt \
        --output bleu_results.json
"""

import argparse
import json
import os
import torch
import sentencepiece as spm
import sacrebleu
from typing import List, Dict
import pandas as pd

# ============ CONSTANTS ============

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_LEN = 80
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}

# ============ MODEL ============

import torch.nn as nn


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

    def forward(self, src, tgt, tgt_mask=None,
                src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        enc_out = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        dec_out = self.decoder(
            tgt_emb, enc_out,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )
        return self.output_projection(dec_out)


# ============ TRANSLATOR ============

class Translator:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading checkpoint from {model_path}...")
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)

        # Load vocab from checkpoint
        if 'token2id' in ckpt and 'id2token' in ckpt:
            self.token2id = ckpt['token2id']
            self.id2token = ckpt['id2token']
            self.vocab_size = ckpt.get('vocab_size', len(self.token2id))
            print("  ✓ Loaded vocabulary from checkpoint")
        else:
            raise ValueError("Checkpoint missing token2id/id2token")

        # Load model
        if 'model_state_dict' in ckpt:
            state = ckpt['model_state_dict']
            config = ckpt.get('config', {})
        elif 'model_state' in ckpt:
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

        # Load SentencePiece
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        print(f"  ✓ Model loaded! Vocab: {self.vocab_size}")

    def _generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=self.device), diagonal=1)
        return mask.bool()

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        if src_lang not in LANG_TAGS or tgt_lang not in LANG_TAGS:
            raise ValueError(f"Invalid lang. Use: {list(LANG_TAGS.keys())}")

        # Encode source
        pieces = self.sp.encode(str(text), out_type=str)
        src_ids = [self.token2id.get(p, UNK_ID) for p in pieces]
        src_ids = [BOS_ID] + src_ids[:MAX_LEN-2] + [EOS_ID]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        # Decode with language tag
        tgt_tag_id = self.token2id[LANG_TAGS[tgt_lang]]
        tgt_ids = [BOS_ID, tgt_tag_id]

        with torch.no_grad():
            for _ in range(MAX_LEN):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=self.device)
                tgt_mask = self._generate_square_subsequent_mask(tgt_tensor.size(1))
                logits = self.model(src_tensor, tgt_tensor, tgt_mask=tgt_mask)
                next_id = logits[0, -1].argmax().item()
                if next_id == EOS_ID:
                    break
                tgt_ids.append(next_id)

        # Decode output
        pieces = []
        for i in tgt_ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            token = self.id2token.get(i, "<unk>")
            if token in LANG_TAGS.values():
                continue
            pieces.append(token)

        return self.sp.decode(pieces)


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
    print(f"\n    Examples (from test set):")
    for s, h, r in zip(sources[:3], hypotheses[:3], references[:3]):
        print(f"      SRC [{src_lang}]: {s[:80]}")
        print(f"      HYP [{tgt_lang}]: {h[:80]}")
        print(f"      REF [{tgt_lang}]: {r[:80]}")
        print()

    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "num_sentences": len(sources),
    }


def main():
    parser = argparse.ArgumentParser(
        description="BLEU Evaluation on Held-Out Test Set (80/10/10 split)"
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="output/spm_unified_multilingual.model")
    parser.add_argument("--model-dir", default="output/phase_5_milestone_3_1",
                        help="Path to model output directory containing test splits")
    parser.add_argument("--output", default="bleu_results.json")
    args = parser.parse_args()

    print("=" * 70)
    print("BLEU EVALUATION — Held-Out Test Set (10% of original data)")
    print("=" * 70)
    print("\n✓ This evaluates on data that was NEVER used during training.")
    print("✓ Scores here reflect TRUE generalization ability on unseen data.\n")

    # Load model
    print(f"[1/3] Loading model from {args.model}...")
    translator = Translator(args.model, args.tokenizer)

    # Load test data from model output directory
    print(f"\n[2/3] Loading held-out test data from {args.model_dir}...")

    tri_test = pd.read_csv(f'{args.model_dir}/test_trilingual.csv')
    en_am_test = pd.read_csv(f'{args.model_dir}/test_en_am.csv')
    am_or_test = pd.read_csv(f'{args.model_dir}/test_am_or.csv')

    print(f"  Trilingual test: {len(tri_test)} sentences")
    print(f"  EN-AM test:      {len(en_am_test)} sentences")
    print(f"  AM-OR test:      {len(am_or_test)} sentences")

    # Evaluate all 6 directions
    print("\n[3/3] Evaluating all 6 translation directions on HELD-OUT test set...")

    results = {}

    # Use trilingual test for all directions
    sources_en = tri_test['English'].astype(str).tolist()
    sources_am = tri_test['Amharic'].astype(str).tolist()
    sources_or = tri_test['Oromo'].astype(str).tolist()

    # EN → AM (use trilingual test)
    results["en_am"] = evaluate_direction(translator, sources_en, sources_am, "en", "am")
    results["en_am"]["source"] = "trilingual_test"

    # AM → EN
    results["am_en"] = evaluate_direction(translator, sources_am, sources_en, "am", "en")
    results["am_en"]["source"] = "trilingual_test"

    # AM → OR
    results["am_or"] = evaluate_direction(translator, sources_am, sources_or, "am", "or")
    results["am_or"]["source"] = "trilingual_test"

    # OR → AM
    results["or_am"] = evaluate_direction(translator, sources_or, sources_am, "or", "am")
    results["or_am"]["source"] = "trilingual_test"

    # EN → OR
    results["en_or"] = evaluate_direction(translator, sources_en, sources_or, "en", "or")
    results["en_or"]["source"] = "trilingual_test"

    # OR → EN
    results["or_en"] = evaluate_direction(translator, sources_or, sources_en, "or", "en")
    results["or_en"]["source"] = "trilingual_test"

    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY — HELD-OUT TEST SET")
    print("=" * 70)

    for direction, metrics in results.items():
        print(f"\n  {direction}:")
        print(f"    BLEU:  {metrics['bleu']:.2f}")
        print(f"    chrF:  {metrics['chrf']:.2f}")
        print(f"    Sents: {metrics['num_sentences']} (held-out, unseen during training)")

    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to {args.output}")

    # Average BLEU
    avg_bleu = sum(m['bleu'] for m in results.values()) / len(results)
    print(f"\n📊 Average BLEU across 6 directions: {avg_bleu:.2f}")
    print("\n" + "=" * 70)
    print("NOTE: These are HONEST scores on truly unseen test data.")
    print("=" * 70)


if __name__ == "__main__":
    main()
