import argparse
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

LANGUAGE_ALIASES = {
    "amharic": ["amharic", "amh", "አማርኛ"],
    "english": ["english", "eng", "en"],
    "oromo": ["oromo", "afan oromo", "afaan oromo", "oromoo", "ormo", "om"]
}

HOMOPHONE_MAP = {
    "ሐ": "ሀ", "ሑ": "ሁ", "ሒ": "ሂ", "ሓ": "ሃ", "ሔ": "ሄ", "ሕ": "ህ", "ሖ": "ሆ",
    "ኀ": "ሀ", "ኁ": "ሁ", "ኂ": "ሂ", "ኃ": "ሃ", "ኄ": "ሄ", "ኅ": "ህ", "ኆ": "ሆ",
    "ኸ": "ሀ", "ኹ": "ሁ", "ኺ": "ሂ", "ኻ": "ሃ", "ኼ": "ሄ", "ኽ": "ህ", "ኾ": "ሆ",
    "ሠ": "ሰ", "ሡ": "ሱ", "ሢ": "ሲ", "ሣ": "ሳ", "ሤ": "ሴ", "ሥ": "ስ", "ሦ": "ሶ",
    "ጸ": "ፀ", "ጹ": "ፁ", "ጺ": "ፂ", "ጻ": "ፃ", "ጼ": "ፄ", "ጽ": "ፅ", "ጾ": "ፆ",
    "ዐ": "አ", "ዑ": "ኡ", "ዒ": "ኢ", "ዓ": "ኣ", "ዔ": "ኤ", "ዕ": "እ", "ዖ": "ኦ"
}

SENTENCE_END_PATTERN = re.compile(r"[።?.!]+$", flags=re.UNICODE)
AMHARIC_CHAR_PATTERN = re.compile(r"[ሀ-፿]", flags=re.UNICODE)
ENGLISH_CHAR_PATTERN = re.compile(r"[A-Za-z]", flags=re.UNICODE)


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_language_for_column(name: str) -> Optional[str]:
    normalized = normalize_column_name(name)
    for lang, aliases in LANGUAGE_ALIASES.items():
        for alias in aliases:
            if alias.replace(" ", "") in normalized:
                return lang
    return None


def infer_columns_from_headers(columns: List[str]) -> Dict[str, str]:
    mapping = {}
    for col in columns:
        lang = find_language_for_column(col)
        if lang and lang not in mapping:
            mapping[lang] = col
    return mapping


def guess_language_from_text(values: List[str]) -> Tuple[float, float, float]:
    text = " ".join([str(v) for v in values if pd.notna(v)])
    amharic_score = len(AMHARIC_CHAR_PATTERN.findall(text))
    english_score = len(ENGLISH_CHAR_PATTERN.findall(text))
    oromo_score = english_score  # Oromo uses Latin script too, so treat as same base
    return float(amharic_score), float(english_score), float(oromo_score)


def guess_columns_by_content(df: pd.DataFrame) -> Dict[str, str]:
    candidates = {col: guess_language_from_text(df[col].head(20).astype(str).tolist()) for col in df.columns}
    guessed = {}
    used_confidence = set()

    # Amharic is easy to distinguish by script.
    for col, (am, en, or_) in candidates.items():
        if am > 0 and am > en:
            guessed["amharic"] = col
            used_confidence.add(col)
            break

    return guessed


def load_dataset(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if path.lower().endswith(('.xls', '.xlsx')):
        return pd.read_excel(path, engine='openpyxl', dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported input format: {path}")


def remove_outer_quotes(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = text.strip()
    if len(text) >= 2 and ((text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'"))):
        return text[1:-1].strip()
    return text


def standardize_punctuation(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFC', text)
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{3,}", "...", text)
    return text


def normalize_amharic(text: str) -> str:
    if not isinstance(text, str):
        return text
    for old, new in HOMOPHONE_MAP.items():
        text = text.replace(old, new)
    text = standardize_punctuation(text)
    text = re.sub(SENTENCE_END_PATTERN, '', text).strip()
    return text


def normalize_latin(text: str, lower: bool = False) -> str:
    if not isinstance(text, str):
        return text
    text = remove_outer_quotes(text)
    text = standardize_punctuation(text)
    if lower:
        text = text.lower()
    return text


def standardize_sentence_punctuation(amharic: str, english: str, oromo: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    amharic = re.sub(SENTENCE_END_PATTERN, '', str(amharic).strip())
    english = re.sub(SENTENCE_END_PATTERN, '', str(english).strip())
    oromo = re.sub(SENTENCE_END_PATTERN, '', str(oromo).strip()) if oromo is not None else None
    if english.endswith('?'):
        amharic = amharic + '?'
        english = english + '?'
        if oromo is not None:
            oromo = oromo + '?'
    else:
        amharic = amharic + '።'
        english = english + '.'
        if oromo is not None:
            oromo = oromo + '.'
    return amharic, english, oromo


def reorder_english_rows(df: pd.DataFrame, amharic_col: str, english_col: str) -> pd.DataFrame:
    rows = []
    i = 0
    while i < len(df):
        row = df.iloc[i].copy()
        amharic = str(row.get(amharic_col, '') or '').strip()
        english = str(row.get(english_col, '') or '').strip()

        if not amharic and english and i + 1 < len(df):
            next_row = df.iloc[i + 1].copy()
            next_amharic = str(next_row.get(amharic_col, '') or '').strip()
            next_english = str(next_row.get(english_col, '') or '').strip()
            if next_amharic:
                next_row[english_col] = f"{english} {next_english}".strip()
                rows.append(next_row)
                i += 2
                continue

        rows.append(row)
        i += 1
    return pd.DataFrame(rows).reset_index(drop=True)


def clean_dataset(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    if 'amharic' in mapping:
        df[mapping['amharic']] = df[mapping['amharic']].apply(normalize_amharic)

    # Normalize all non-Amharic columns with Latin-style cleanup.
    for col in df.columns:
        if 'amharic' in mapping and col == mapping['amharic']:
            continue
        lower = 'english' in mapping and col == mapping['english']
        df[col] = df[col].apply(lambda text: normalize_latin(text, lower=lower))

    if 'amharic' in mapping and 'english' in mapping:
        if 'oromo' in mapping:
            df[[mapping['amharic'], mapping['english'], mapping['oromo']]] = df.apply(
                lambda row: standardize_sentence_punctuation(
                    row[mapping['amharic']], row[mapping['english']], row[mapping['oromo']]
                ),
                axis=1,
                result_type='expand'
            )
        else:
            df[[mapping['amharic'], mapping['english']]] = df.apply(
                lambda row: standardize_sentence_punctuation(
                    row[mapping['amharic']], row[mapping['english']], None
                )[:2],
                axis=1,
                result_type='expand'
            )

    df = df.replace({'': None}).dropna(how='all').reset_index(drop=True)
    return df


def build_column_mapping(df: pd.DataFrame, explicit_cols: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    columns = list(df.columns)
    mapping = explicit_cols.copy() if explicit_cols else {}
    inferred = infer_columns_from_headers(columns)
    for lang, col in inferred.items():
        if lang not in mapping:
            mapping[lang] = col
    if 'amharic' not in mapping or 'english' not in mapping:
        content_guess = guess_columns_by_content(df)
        for lang, col in content_guess.items():
            if lang not in mapping:
                mapping[lang] = col
    return mapping


def parse_column_mapping(mapping_arg: Optional[str]) -> Dict[str, str]:
    if not mapping_arg:
        return {}
    mapping = {}
    for pair in mapping_arg.split(','):
        if '=' not in pair:
            raise argparse.ArgumentTypeError("Columns must be provided as Lang=ColumnName pairs")
        lang, col = pair.split('=', 1)
        lang = lang.strip().lower()
        if lang not in LANGUAGE_ALIASES:
            raise argparse.ArgumentTypeError(f"Unknown language key: {lang}")
        mapping[lang] = col.strip()
    return mapping


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sep = '\t'
    df.to_csv(output_path, index=False, sep=sep)


def process_file(input_path: str, output_path: Optional[str], column_mapping: Optional[Dict[str, str]], reorder_english: bool) -> None:
    print(f"Processing: {input_path}")
    df = load_dataset(input_path)
    mapping = build_column_mapping(df, explicit_cols=column_mapping)

    if 'amharic' in mapping and 'english' in mapping and reorder_english:
        df = reorder_english_rows(df, mapping['amharic'], mapping['english'])

    cleaned = clean_dataset(df, mapping)
    out_path = output_path or os.path.join('output', 'cleaned_dataset', f"normalized_{os.path.splitext(os.path.basename(input_path))[0]}.csv")
    save_dataset(cleaned, out_path)
    print(f"Saved normalized file to: {out_path}")
    print(f"Rows in cleaned file: {len(cleaned)}")
    print("Mapping used:")
    for lang, col in mapping.items():
        print(f"  {lang}: {col}")
    print("-" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description='General dataset normalizer for Amharic, English, and Oromo datasets.')
    parser.add_argument('-i', '--input', help='Input CSV or Excel file to normalize.')
    parser.add_argument('-d', '--input-dir', help='Directory containing dataset files to normalize.')
    parser.add_argument('-o', '--output', help='Output file path. If omitted, writes to output/cleaned_dataset/normalized_<name>.csv')
    parser.add_argument('--columns', help='Explicit language column mapping, e.g. Amharic=Col1,English=Col2,Oromo=Col3')
    parser.add_argument('--no-reorder', action='store_true', help='Disable English-before-Amharic row reorder fix.')
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error('Either --input or --input-dir is required.')

    mapping = parse_column_mapping(args.columns) if args.columns else {}
    files = []
    if args.input:
        files.append((args.input, args.output))
    if args.input_dir:
        for name in sorted(os.listdir(args.input_dir)):
            if name.lower().endswith(('.csv', '.tsv', '.xls', '.xlsx')):
                in_path = os.path.join(args.input_dir, name)
                out_path = None
                if args.output and os.path.isdir(args.output):
                    out_path = os.path.join(args.output, f"normalized_{os.path.splitext(name)[0]}.csv")
                files.append((in_path, out_path))

    for input_path, output_path in files:
        process_file(input_path, output_path, mapping, not args.no_reorder)


if __name__ == '__main__':
    main()
