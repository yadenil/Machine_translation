#!/usr/bin/env python3
"""
Machine Translation Inference Script — Phase 5 Final Model

Translate text using the trained multilingual model (English ↔ Oromo ↔ Amharic).
Supports both batch inference and interactive mode.
Compatible with Phase 5 checkpoint formats (Milestone 2.2, 3.1).

Usage:
    # Interactive mode
    python inference_final.py

    # Batch inference from file
    python inference_final.py --input input.txt --source en --target or --output output.txt

    # Single translation
    python inference_final.py --text "Hello world" --source en --target or

    # Use specific checkpoint
    python inference_final.py --model output/phase_5_milestone_3_1/final_translator_multilingual.pt --text "Hello" --source en --target am
"""

import argparse
import os
import sys
import json
import torch
import torch.nn as nn
import sentencepiece as spm
from pathlib import Path
from typing import List, Optional

# Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special token IDs (must match training)
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
MAX_LEN = 80

# Language tags (must match training tokenizer setup)
LANG_TAGS = {"am": "<am>", "or": "<or>", "en": "<en>"}


# ============ MODEL DEFINITION (must match training) ============

class SimpleTransformer(nn.Module):
    """Same architecture as Phase 5 training scripts."""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512, 
            batch_first=True, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=512,
            batch_first=True, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt):
        src_emb = self.embedding(src)
        tgt_emb = self.embedding(tgt)
        enc_out = self.encoder(src_emb)
        dec_out = self.decoder(tgt_emb, enc_out)
        return self.output_projection(dec_out)


# ============ TRANSLATION MODEL WRAPPER ============

class TranslationModel:
    """Wrapper for inference with Phase 5 trained models."""

    def __init__(self, model_path: str, tokenizer_path: str = "output/spm_unified_multilingual.model"):
        """
        Load model and tokenizer.

        Args:
            model_path: Path to saved model checkpoint (Phase 5 format)
            tokenizer_path: Path to SentencePiece tokenizer
        """
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        print(f"Loading model from {model_path}...")
        
        # Load checkpoint
        ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
        
        # Handle different checkpoint formats
        self.model_state, self.model_config = self._parse_checkpoint(ckpt)
        
        # Build token2id / id2token from tokenizer
        self._build_vocab(tokenizer_path)
        
        # Create and load model
        self.model = SimpleTransformer(
            self.vocab_size,
            d_model=self.model_config.get('d_model', 128),
            n_heads=self.model_config.get('n_heads', 4),
            n_layers=self.model_config.get('n_layers', 2),
        ).to(DEVICE)
        
        self.model.load_state_dict(self.model_state)
        self.model.eval()
        
        # Load SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {DEVICE}")
        print(f"   Vocabulary size: {self.vocab_size:,}")
        print(f"   Model config: {self.model_config}")

    def _parse_checkpoint(self, ckpt):
        """Parse different Phase 5 checkpoint formats."""
        
        # Format 1: Standard PyTorch with model_state_dict + config
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
                config = ckpt.get('config', {})
                print("  ✓ Loaded standard checkpoint (model_state_dict + config)")
                return state, config
            
            # Format 2: Custom format from original training (model_state, vocab_size, etc.)
            elif 'model_state' in ckpt:
                state = ckpt['model_state']
                config = ckpt.get('config', {})
                print("  ✓ Loaded custom checkpoint (model_state + config)")
                return state, config
            
            # Format 3: Just the state dict directly
            else:
                # Try to detect if it's a raw state dict
                sample_key = list(ckpt.keys())[0]
                if 'encoder' in sample_key or 'embedding' in sample_key:
                    print("  ✓ Loaded raw state_dict")
                    return ckpt, {}
                else:
                    raise ValueError(f"Unknown checkpoint format. Keys: {list(ckpt.keys())[:10]}")
        else:
            raise ValueError("Checkpoint is not a dictionary")

    def _build_vocab(self, tokenizer_path: str):
        """Build token2id and id2token from SentencePiece model."""
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_path)
        
        self.vocab_size = sp.GetPieceSize()
        
        # Build mappings
        self.token2id = {}
        self.id2token = {}
        
        for i in range(self.vocab_size):
            piece = sp.IdToPiece(i)
            self.token2id[piece] = i
            self.id2token[i] = piece
        
        # Add special tokens if not present
        specials = {
            '<pad>': PAD_ID,
            '<unk>': UNK_ID,
            '<s>': BOS_ID,
            '</s>': EOS_ID,
            '<am>': self.vocab_size,      # Will be added if not in vocab
            '<or>': self.vocab_size + 1,
            '<en>': self.vocab_size + 2,
        }
        
        # Check if language tags exist in tokenizer
        for tag in ['<am>', '<or>', '<en>']:
            if tag in self.token2id:
                print(f"  ✓ Language tag '{tag}' found in tokenizer (id={self.token2id[tag]})")
            else:
                print(f"  ⚠ Language tag '{tag}' NOT in tokenizer. Translation may fail.")
        
        print(f"  ✓ Vocabulary built: {len(self.token2id)} tokens")

    def _encode(self, text: str) -> List[int]:
        """Encode text to token IDs using SentencePiece."""
        pieces = self.sp.encode(str(text), out_type=str)
        ids = [self.token2id.get(p, UNK_ID) for p in pieces]
        ids = ids[: MAX_LEN - 2]  # Leave room for BOS and EOS
        return [BOS_ID] + ids + [EOS_ID]

    def _decode(self, ids: List[int]) -> str:
        """Decode token IDs to text using SentencePiece."""
        # Filter out special tokens
        valid_ids = []
        for i in ids:
            if i in (PAD_ID, BOS_ID, EOS_ID):
                continue
            if i in self.id2token:
                piece = self.id2token[i]
                # Skip language tags in output
                if piece in ('<am>', '<or>', '<en>'):
                    continue
                valid_ids.append(i)
        
        return self.sp.decode(valid_ids)

    def translate(self, text: str, source_lang: str, target_lang: str, beam_size: int = 1) -> str:
        """
        Translate text from source to target language.

        Args:
            text: Input text to translate
            source_lang: Source language code ("en", "or", "am")
            target_lang: Target language code ("en", "or", "am")
            beam_size: 1 for greedy, >1 for beam search (not implemented yet)

        Returns:
            Translated text
        """
        if source_lang not in LANG_TAGS or target_lang not in LANG_TAGS:
            raise ValueError(
                f"Invalid language codes. Supported: {', '.join(LANG_TAGS.keys())}"
            )

        # Encode source text
        src_ids = self._encode(text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

        # Prepare decoder input: BOS + target language tag
        tgt_tag = LANG_TAGS[target_lang]
        tgt_tag_id = self.token2id.get(tgt_tag, UNK_ID)
        
        # Start with BOS + language tag
        tgt_ids = [BOS_ID, tgt_tag_id]

        # Greedy decoding
        with torch.no_grad():
            for _ in range(MAX_LEN):
                tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long, device=DEVICE)
                logits = self.model(src_tensor, tgt_tensor)
                
                # Get next token (last position)
                next_id = logits[0, -1].argmax().item()

                if next_id == EOS_ID:
                    break

                tgt_ids.append(next_id)

        return self._decode(tgt_ids)

    def translate_batch(self, texts: List[str], source_lang: str, target_lang: str) -> List[str]:
        """
        Translate multiple texts.

        Args:
            texts: List of input texts
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            List of translated texts
        """
        results = []
        for i, text in enumerate(texts):
            try:
                translation = self.translate(text, source_lang, target_lang)
                results.append(translation)
                if (i + 1) % 100 == 0:
                    print(f"  Translated {i+1}/{len(texts)}...")
            except Exception as e:
                print(f"Warning: Error translating '{text[:50]}...': {e}")
                results.append("")
        
        return results


# ============ INTERACTIVE MODE ============

def interactive_mode(model: TranslationModel):
    """Interactive translation mode."""
    print("\n" + "=" * 80)
    print("INTERACTIVE TRANSLATION MODE — Phase 5 Final Model")
    print("=" * 80)
    print("\nSupported languages:")
    print("  en - English")
    print("  or - Oromo")
    print("  am - Amharic")
    print("\nType 'quit' or 'exit' to end.")
    print("Type 'switch' to change language pair.\n")

    src_lang = "en"
    tgt_lang = "or"

    while True:
        try:
            user_input = input(f"[{src_lang}→{tgt_lang}] > ").strip().lower()

            if user_input in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            if user_input == "switch":
                src_lang = input("Source language (en/or/am): ").strip().lower()
                tgt_lang = input("Target language (en/or/am): ").strip().lower()
                if src_lang not in LANG_TAGS or tgt_lang not in LANG_TAGS:
                    print("Invalid codes. Using default en→or")
                    src_lang, tgt_lang = "en", "or"
                continue

            if not user_input:
                continue

            print("Translating...")
            result = model.translate(user_input, src_lang, tgt_lang)

            lang_names = {"en": "English", "or": "Oromo", "am": "Amharic"}
            print(f"\n  [{lang_names[src_lang]} → {lang_names[tgt_lang]}]")
            print(f"  Source: {user_input}")
            print(f"  Target: {result}\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


# ============ BATCH INFERENCE ============

def batch_inference(
    model: TranslationModel,
    input_file: str,
    source_lang: str,
    target_lang: str,
    output_file: Optional[str] = None,
    encoding: str = "utf-8",
):
    """Translate all sentences from input file."""
    with open(input_file, "r", encoding=encoding, errors="replace") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    print(f"\nTranslating {len(texts)} sentences...")
    print(f"Source: {source_lang} → Target: {target_lang}")

    translations = model.translate_batch(texts, source_lang, target_lang)

    # Save
    if output_file:
        with open(output_file, "w", encoding=encoding) as f:
            for t in translations:
                f.write(t + "\n")
        print(f"\n✅ Saved to: {output_file}")

    # Summary
    successful = sum(1 for t in translations if t)
    print(f"✅ Successful: {successful}/{len(texts)}")

    # Show first 5
    print("\nFirst 5 translations:")
    print("-" * 80)
    for src, tgt in zip(texts[:5], translations[:5]):
        print(f"  {src[:60]}...")
        print(f"  → {tgt[:60]}...")
        print()

    return translations


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(
        description="Translate with Phase 5 Final Multilingual Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python inference_final.py

  # Single translation
  python inference_final.py --text "Hello world" --source en --target or

  # Batch from file
  python inference_final.py --input input.txt --source en --target am --output out.txt

  # Use specific checkpoint
  python inference_final.py --model output/phase_5_milestone_3_1/final_translator_best.pt --text "Test" --source en --target or
        """,
    )

    parser.add_argument("--text", type=str, help="Single text to translate")
    parser.add_argument("--input", type=str, help="Input file (one sentence per line)")
    parser.add_argument("--output", type=str, help="Output file")
    parser.add_argument("--source", type=str, default="en", help="Source: en/or/am")
    parser.add_argument("--target", type=str, default="or", help="Target: en/or/am")
    parser.add_argument("--model", type=str, 
                        default="output/phase_5_milestone_3_1/final_translator_multilingual.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str,
                        default="output/spm_unified_multilingual.model",
                        help="Path to SentencePiece tokenizer")
    parser.add_argument("--encoding", type=str, default="utf-8")

    args = parser.parse_args()

    # Load model
    try:
        model = TranslationModel(args.model, args.tokenizer)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # Route
    if args.text:
        try:
            result = model.translate(args.text, args.source, args.target)
            lang_names = {"en": "English", "or": "Oromo", "am": "Amharic"}
            print(f"\n[{lang_names[args.source]} → {lang_names[args.target]}]")
            print(f"  {args.text}")
            print(f"  → {result}\n")
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    elif args.input:
        try:
            batch_inference(model, args.input, args.source, args.target, args.output, args.encoding)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)

    else:
        interactive_mode(model)


if __name__ == "__main__":
    main()