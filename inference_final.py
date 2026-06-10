#!/usr/bin/env python3
"""
Inference with Language Tags — Phase 5 Final Model
"""

import argparse
import os
import torch
import torch.nn as nn
import sentencepiece as spm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


class TranslationModel:
    def __init__(self, model_path: str, tokenizer_path: str = "output/spm_unified_multilingual.model"):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        print(f"Loading model from {model_path}...")
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Load vocab from checkpoint
        if 'token2id' in ckpt and 'id2token' in ckpt:
            self.token2id = ckpt['token2id']
            self.id2token = ckpt['id2token']
            self.vocab_size = ckpt.get('vocab_size', len(self.token2id))
            print("✓ Loaded vocabulary from checkpoint")
        else:
            raise ValueError("Checkpoint missing token2id/id2token. Use original training script.")
        
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
        ).to(DEVICE)
        
        self.model.load_state_dict(state)
        self.model.eval()
        
        # Load SentencePiece for decode
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        print(f"✅ Model loaded! Vocab: {self.vocab_size}")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if source_lang not in LANG_TAGS or target_lang not in LANG_TAGS:
            raise ValueError(f"Invalid lang. Use: {list(LANG_TAGS.keys())}")

        # Encode source
        pieces = self.sp.encode(str(text), out_type=str)
        src_ids = [self.token2id.get(p, UNK_ID) for p in pieces]
        src_ids = [BOS_ID] + src_ids[:MAX_LEN-2] + [EOS_ID]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

        # Decode with language tag
        tgt_tag_id = self.token2id[LANG_TAGS[target_lang]]
        tgt_ids = [BOS_ID, tgt_tag_id]

        with torch.no_grad():
            for _ in range(MAX_LEN):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
                logits = self.model(src_tensor, tgt_tensor)
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

    def translate_batch(self, texts, source_lang, target_lang):
        return [self.translate(t, source_lang, target_lang) for t in texts]


def interactive_mode(model):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Commands: 'switch' to change languages, 'quit' to exit\n")

    src_lang, tgt_lang = "en", "or"

    while True:
        try:
            user_input = input(f"[{src_lang}→{tgt_lang}] > ").strip().lower()
            if user_input in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if user_input == "switch":
                src_lang = input("Source (en/or/am): ").strip().lower()
                tgt_lang = input("Target (en/or/am): ").strip().lower()
                continue
            if not user_input:
                continue

            result = model.translate(user_input, src_lang, tgt_lang)
            print(f"  → {result}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    parser.add_argument("--source", type=str, default="en")
    parser.add_argument("--target", type=str, default="or")
    parser.add_argument("--model", type=str,
                        default="output/phase_5_milestone_3_1/final_translator_best.pt")
    parser.add_argument("--tokenizer", type=str,
                        default="output/spm_unified_multilingual.model")
    args = parser.parse_args()

    model = TranslationModel(args.model, args.tokenizer)

    if args.text:
        result = model.translate(args.text, args.source, args.target)
        print(f"{args.text} → {result}")
    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()