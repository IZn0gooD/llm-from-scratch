"""
Phase 1 -- genere du texte a partir d'un checkpoint entraine.

Usage:
    python sample.py --ckpt checkpoints/best.pt --prompt "The history of" --max_new_tokens 200
"""
import argparse
import os
import sys

import tiktoken
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # le BPE GPT-2 peut decoder vers n'importe quel caractere Unicode ; la codepage Windows par defaut (cp1252) plante sinon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelConfig, ToyLLM


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="The history of")
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    config = ModelConfig(**ckpt["config"])
    model = ToyLLM(config).to(args.device)

    state_dict = ckpt["model"]
    unwanted_prefix = "_orig_mod."  # present si le checkpoint vient d'un model torch.compile()
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()

    val_loss = ckpt.get("best_val_loss")
    val_loss_str = f"{val_loss:.4f}" if val_loss is not None else "n/a"
    print(f"checkpoint charge : step={ckpt.get('step')}, best_val_loss={val_loss_str}")

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode_ordinary(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    with torch.no_grad():
        y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)

    print(enc.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
