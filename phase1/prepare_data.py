"""
Phase 1 -- streame FineWeb-Edu (sample-10BT) depuis HuggingFace et tokenise en shards .bin
(uint16, tokenizer GPT-2/tiktoken). Pas de telechargement complet prealable : le dataset est
lu en streaming (datasets.load_dataset(..., streaming=True)) et tokenise au fil de l'eau.

Necessite : pip install datasets

Usage typique (run complet, ~10B tokens, shards de 100M tokens ; le shard 0 sert de val) :
    python prepare_data.py --out_dir data

Test rapide (quelques shards seulement) :
    python prepare_data.py --out_dir data_smoke --max_tokens 2_000_000 --shard_size 500_000
"""
import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceFW/fineweb-edu"
DATASET_CONFIG = "sample-10BT"


def colab_drive_dir(subdir):
    """Sur Google Colab, monte le Drive (si pas deja monte) et retourne un chemin persistant dessus.
    Hors Colab, retourne None (l'appelant utilise alors un chemin local)."""
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return None
    if not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    base = "/content/drive/MyDrive/llm-phase1"
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, subdir)


def tokenize_doc(enc, eot, text):
    ids = enc.encode_ordinary(text)
    ids.append(eot)
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out_dir = colab_drive_dir("data") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ap.add_argument("--out_dir", default=default_out_dir, help="par defaut : local hors Colab, sinon /content/drive/MyDrive/llm-phase1/data (Drive monte automatiquement)")
    ap.add_argument("--shard_size", type=int, default=100_000_000, help="tokens par shard (defaut 100M, ~200MB en uint16)")
    ap.add_argument("--max_tokens", type=int, default=None, help="plafonne le nb total de tokens tokenises (test rapide)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    print(f"tokenizer: gpt2 (tiktoken), vocab_size={enc.n_vocab}, eot={eot}")
    print(f"dataset: {DATASET_NAME} (config={DATASET_CONFIG}), streaming=True")

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train", streaming=True)

    shard_index = 0
    buffer = np.empty((args.shard_size,), dtype=np.uint16)
    pos = 0
    total_tokens = 0

    def flush(n, idx):
        split = "val" if idx == 0 else "train"
        path = os.path.join(args.out_dir, f"{split}_{idx:06d}.bin")
        buffer[:n].tofile(path)
        print(f"shard {idx:06d} ({split}): {n:,} tokens -> {path}")

    pbar = tqdm(unit="tok", unit_scale=True, desc="tokenize FineWeb-Edu")
    stop = False
    for doc in ds:
        ids = tokenize_doc(enc, eot, doc["text"])
        i = 0
        while i < len(ids):
            space = args.shard_size - pos
            take = min(space, len(ids) - i)
            buffer[pos:pos + take] = ids[i:i + take]
            pos += take
            i += take
            total_tokens += take
            pbar.update(take)
            if pos == args.shard_size:
                flush(pos, shard_index)
                shard_index += 1
                pos = 0
            if args.max_tokens and total_tokens >= args.max_tokens:
                stop = True
                break
        if stop:
            break
    if pos > 0:
        flush(pos, shard_index)
        shard_index += 1
    pbar.close()
    print(f"termine: {total_tokens:,} tokens au total, {shard_index} shards ecrits dans {args.out_dir}")


if __name__ == "__main__":
    main()
