"""
Phase 0 -- telecharge TinyStoriesV2-GPT4 (train+valid) et tokenise en .bin (uint16, tokenizer GPT-2/tiktoken).

Usage typique (run complet) :
    python prepare_data.py

Test rapide (ne telecharge/tokenise qu'un echantillon) :
    python prepare_data.py --out_dir data_smoke --max_download_mb 5 --limit_train_stories 300 --limit_val_stories 100
"""
import argparse
import os
import urllib.request

import numpy as np
import tiktoken
from tqdm import tqdm

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
FILES = {
    "train": "TinyStoriesV2-GPT4-train.txt",
    "val": "TinyStoriesV2-GPT4-valid.txt",
}
STORY_SEP = "<|endoftext|>"


def download(url: str, dest: str, max_bytes: int = None, force: bool = False):
    if os.path.exists(dest) and not force:
        print(f"[skip] {dest} existe deja (--force pour re-telecharger)")
        return
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "phase0-prepare/1.0"})
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0)) or None
        if max_bytes and total:
            total = min(total, max_bytes)
        block = 1 << 20  # 1 MiB
        written = 0
        with open(tmp, "wb") as out, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(dest)) as pbar:
            while True:
                chunk = resp.read(block)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                pbar.update(len(chunk))
                if max_bytes and written >= max_bytes:
                    break
    os.replace(tmp, dest)


def iter_stories(path: str, limit: int = None):
    """Genere le texte de chaque histoire, en streamant le fichier ligne par ligne (pas de chargement complet en RAM)."""
    buf = []
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == STORY_SEP:
                story = "".join(buf).strip()
                buf = []
                if story:
                    yield story
                    n += 1
                    if limit and n >= limit:
                        return
            else:
                buf.append(line)
    story = "".join(buf).strip()
    if story:
        yield story


def tokenize_split(txt_path: str, out_path: str, enc, eot: int, limit: int = None, flush_every: int = 2000) -> int:
    buffer = []
    total_tokens = 0
    with open(out_path, "wb") as out:
        for i, story in enumerate(tqdm(iter_stories(txt_path, limit=limit), desc=f"tokenize -> {os.path.basename(out_path)}", unit="histoire"), 1):
            ids = enc.encode_ordinary(story)
            ids.append(eot)
            buffer.extend(ids)
            if i % flush_every == 0:
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(out)
                total_tokens += len(buffer)
                buffer = []
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(out)
            total_tokens += len(buffer)
    return total_tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--limit_train_stories", type=int, default=None, help="limite le nb d'histoires train (test rapide)")
    ap.add_argument("--limit_val_stories", type=int, default=None, help="limite le nb d'histoires val (test rapide)")
    ap.add_argument("--max_download_mb", type=float, default=None, help="plafonne le telechargement du fichier train (test rapide)")
    ap.add_argument("--force", action="store_true", help="re-telecharge meme si le fichier brut existe deja")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_dir = os.path.join(args.out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token
    print(f"tokenizer: gpt2 (tiktoken), vocab_size={enc.n_vocab}, eot={eot}")

    max_train_bytes = int(args.max_download_mb * (1 << 20)) if args.max_download_mb else None

    for split, fname in FILES.items():
        raw_path = os.path.join(raw_dir, fname)
        cap = max_train_bytes if split == "train" else None
        download(BASE_URL + fname, raw_path, max_bytes=cap, force=args.force)

    for split, fname in FILES.items():
        raw_path = os.path.join(raw_dir, fname)
        out_path = os.path.join(args.out_dir, f"{split}.bin")
        limit = args.limit_train_stories if split == "train" else args.limit_val_stories
        n_tokens = tokenize_split(raw_path, out_path, enc, eot, limit=limit)
        size_mb = os.path.getsize(out_path) / (1 << 20)
        print(f"{split}: {n_tokens:,} tokens -> {out_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
