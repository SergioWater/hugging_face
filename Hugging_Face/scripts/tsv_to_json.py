#!/usr/bin/env python

"""
Convert a Mozilla Common Voice .tsv file into a .json manifest 
compatible with NeMo or other ASR pipelines.

Usage example:
  python tsv_to_json.py \
    --tsv=cv-corpus-9.0-2022-04-27/rw/train.tsv \
    --folder=cv-corpus-9.0-2022-04-27/rw/clips \
    --sampling_count=-1
"""

import pandas as pd
import json
import tqdm
import argparse
import os

def main():
    parser = argparse.ArgumentParser("MCV TSV-to-JSON converter")
    parser.add_argument("--tsv", required=True, type=str, help="Path to input TSV file (e.g. train.tsv)")
    parser.add_argument("--folder", required=True, type=str, help="Path to folder with audio files (e.g. clips/)")
    parser.add_argument("--sampling_count", required=True, type=int, help="Number of examples you want; use -1 for all")
    args = parser.parse_args()

    if not os.path.exists(args.tsv):
        raise FileNotFoundError(f"TSV file not found at {args.tsv}")
    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f"Audio folder not found at {args.folder}")

    print(f"[DEBUG] Loading TSV: {args.tsv}")
    df = pd.read_csv(args.tsv, sep='\t')

    required_cols = ['client_id', 'path', 'sentence', 'up_votes', 'down_votes']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {args.tsv}")

    # Convert numeric columns safely
    df['up_votes'] = pd.to_numeric(df['up_votes'], errors='coerce').fillna(0).astype(int)
    df['down_votes'] = pd.to_numeric(df['down_votes'], errors='coerce').fillna(0).astype(int)

    output_json = args.tsv.replace('.tsv', '.json')
    print(f"[DEBUG] Output JSON = {output_json}")

    if args.sampling_count > 0 and args.sampling_count < len(df):
        mod = len(df) // args.sampling_count
    else:
        mod = 1

    with open(output_json, 'w', encoding='utf-8') as fo:
        for idx in tqdm.tqdm(range(len(df)), desc="Converting TSV rows"):
            if idx % mod != 0:
                continue

            audio_file = os.path.join(args.folder, df['path'][idx])
            text = df['sentence'][idx]

            item = {
                'audio_filepath': audio_file,
                'text': text,
                'up_votes': int(df['up_votes'][idx]),
                'down_votes': int(df['down_votes'][idx]),
                'age': df['age'][idx] if 'age' in df.columns else None,
                'gender': df['gender'][idx] if 'gender' in df.columns else None,
                'accents': df['accents'][idx] if 'accents' in df.columns else None,
                'client_id': df['client_id'][idx]
            }
            fo.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[INFO] Wrote {output_json}")

if __name__ == "__main__":
    main()
