#!/usr/bin/env python3
"""
Debug Inference — Phase 5 Final Model
Shows exactly what tokens are being processed to fix the 'ibibib' issue.
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
        
        if 'token2id' in ckpt and 'id2token' in ckpt:
            self.token2id = ckpt['token2id']
            # Ensure keys are integers for id2token
            self.id2token = {int(k): v for k, v in ckpt['id2token'].items()}
            self.vocab_size = ckpt.get('vocab_size', len(self.token2id))
            print(f"✓ Loaded vocabulary (Size: {self.vocab_size})")
        else:
            raise ValueError("Checkpoint missing token2id/id2token.")
        
        state = ckpt.get('model_state_dict') or ckpt.get('model_state') or ckpt
        config = ckpt.get('config', {})
        
        self.model = SimpleTransformer(
            self.vocab_size,
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 2),
        ).to(DEVICE)
        
        self.model.load_state_dict(state)
        self.model.eval()
        
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

    def translate(self, text: str, source_lang: str, target_lang: str, debug: bool = True) -> str:
        if source_lang not in LANG_TAGS or target_lang not in LANG_TAGS:
            raise ValueError(f"Invalid lang. Use: {list(LANG_TAGS.keys())}")

        # 1. Encode source
        pieces = self.sp.encode(str(text), out_type=str)
        src_ids = [self.token2id.get(p, UNK_ID) for p in pieces]
        src_ids = [BOS_ID] + src_ids[:MAX_LEN-2] + [EOS_ID]
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

        if debug:
            print(f"\n[DEBUG ENCODE] '{text}'")
            print(f"  Pieces: {pieces}")
            print(f"  IDs:    {src_ids}")

        # 2. Setup Decoder with target language tag
        tgt_tag = LANG_TAGS[target_lang]
        if tgt_tag not in self.token2id:
            return f"ERROR: Tag {tgt_tag} not in vocab!"
            
        tgt_tag_id = self.token2id[tgt_tag]
        tgt_ids = [BOS_ID, tgt_tag_id]

        # 3. Autoregressive Loop
        with torch.no_grad():
            for step in range(MAX_LEN):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
                logits = self.model(src_tensor, tgt_tensor)
                
                # Take argmax of the last token
                next_id = logits[0, -1].argmax().item()
                
                if debug and step < 5: # Only print first few steps to avoid clutter
                    token_str = self.id2token.get(next_id, f"ID:{next_id}")
                    print(f"  Step {step}: Predicted ID {next_id} ('{token_str}')")

                if next_id == EOS_ID:
                    break
                tgt_ids.append(next_id)
                
                if len(tgt_ids) > MAX_LEN:
                    break

        # 4. Decode
        output_pieces = []
        for i in tgt_ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            token = self.id2token.get(i, "??")
            if token in LANG_TAGS.values():
                continue
            output_pieces.append(token)

        return self.sp.decode(output_pieces)

def main():
    # Use defaults similar to your original script
    model_path = "output/phase_5_milestone_3_1/final_translator_best.pt"
    tokenizer_path = "output/spm_unified_multilingual.model"
    
    try:
        model = TranslationModel(model_path, tokenizer_path)
        print("\nDebug mode active. Testing single translation...")
        
        test_text = "hello"
        result = model.translate(test_text, "en", "or")
        print(f"\nFinal Result: {result}")
        
    except Exception as e:
        print(f"Failed to run debug: {e}")

if __name__ == "__main__":
    main()