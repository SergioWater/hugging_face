#!/usr/bin/env python

"""
Resample audio to 16kHz mono from the original .mp3 or other sample rates.

Usage:
  python decode_resample.py \
    --manifest=cv-corpus-9.0-2022-04-27/rw/train.json \
    --destination_folder=./train
    --num_workers=8
"""

import argparse
import os
import json
import sox
from sox import Transformer
import tqdm
import multiprocessing
from tqdm.contrib.concurrent import process_map

DEST_FOLDER = None

def process_line(item):
    if not isinstance(item.get('text', ''), str):
        item['text'] = ''
    else:
        item['text'] = item['text'].lower().strip()

    audio_path = item['audio_filepath']
    if not os.path.exists(audio_path):
        # skip if original file not found
        return None

    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_wav_path = os.path.join(DEST_FOLDER, base_name + ".wav")

    if not os.path.exists(output_wav_path):
        tfm = Transformer()
        tfm.rate(samplerate=16000)
        tfm.channels(n_channels=1)
        tfm.build(input_filepath=audio_path, output_filepath=output_wav_path)

    item['audio_filepath'] = output_wav_path
    item['duration'] = sox.file_info.duration(output_wav_path)
    return item

def load_data(manifest_path):
    data = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def main():
    global DEST_FOLDER

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=str, help='Path to the original .json manifest')
    parser.add_argument('--destination_folder', required=True, type=str, help='Where to write .wav files')
    parser.add_argument('--num_workers', default=multiprocessing.cpu_count(), type=int, help='Workers to process dataset.')
    args = parser.parse_args()

    MANIFEST_PATH = args.manifest
    DEST_FOLDER = args.destination_folder

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_PATH}")
    os.makedirs(DEST_FOLDER, exist_ok=True)

    print(f"[DEBUG] Loading data from {MANIFEST_PATH}")
    data = load_data(MANIFEST_PATH)
    print(f"[DEBUG] Found {len(data)} items in the manifest.")

    print(f"[DEBUG] Resampling & converting to .wav @16k in {DEST_FOLDER} with {args.num_workers} workers...")

    data_new = process_map(process_line, data, max_workers=args.num_workers, chunksize=100)

    data_filtered = [x for x in data_new if x is not None]

    out_path = MANIFEST_PATH.replace('.json', '_decoded.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        for item in tqdm.tqdm(data_filtered, desc="Writing final JSON"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"[INFO] Wrote final manifest to {out_path}")
    print(f"[INFO] Wrote {len(data_filtered)} entries (some might have been missing).")

if __name__ == "__main__":
    main()
