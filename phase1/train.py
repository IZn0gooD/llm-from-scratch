"""
Phase 1 -- boucle d'entrainement complete pour ToyLLM sur les shards .bin produits par prepare_data.py
(FineWeb-Edu sample-10BT). Meme boucle que Phase 0, adaptee pour lire plusieurs shards et une config
de modele plus grande (~320M params par defaut).

Exemple (config par defaut, ~320M params) :
    python train.py --data_dir data --out_dir checkpoints

Reprise apres interruption (essentiel sur Colab : les sessions sont limitees dans le temps) :
    python train.py --data_dir data --out_dir checkpoints --resume
"""
import argparse
import glob
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import tiktoken
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # le BPE GPT-2 peut decoder vers n'importe quel caractere Unicode ; la codepage Windows par defaut (cp1252) plante sinon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import ModelConfig, ToyLLM


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    # donnees / sortie
    ap.add_argument("--data_dir", default=os.path.join(here, "data"))
    ap.add_argument("--out_dir", default=os.path.join(here, "checkpoints"))
    ap.add_argument("--resume", action="store_true", help="reprend depuis out_dir/last.pt s'il existe")
    # architecture (~320M params par defaut : dim=1024, 24 layers, 16 heads / 4 kv_heads, seq 1024)
    ap.add_argument("--vocab_size", type=int, default=50304)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--n_layers", type=int, default=24)
    ap.add_argument("--n_heads", type=int, default=16)
    ap.add_argument("--n_kv_heads", type=int, default=4)
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0)
    # optimisation (batch_size = micro-batch ; grad_accum_steps augmente le batch effectif sans VRAM supplementaire)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum_steps", type=int, default=32)
    ap.add_argument("--max_steps", type=int, default=20000)
    ap.add_argument("--warmup_steps", type=int, default=700)
    ap.add_argument("--max_lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    # boucle
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--eval_interval", type=int, default=500)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--sample_tokens", type=int, default=150)
    # divers
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--compile", action="store_true", help="torch.compile(model) -- OK sur Linux/cloud, support Windows/Triton limite")
    ap.add_argument("--seed", type=int, default=1337)
    return ap.parse_args()


def load_data(data_dir):
    train_paths = sorted(glob.glob(os.path.join(data_dir, "train_*.bin")))
    val_paths = sorted(glob.glob(os.path.join(data_dir, "val_*.bin")))
    if not train_paths or not val_paths:
        raise FileNotFoundError(
            f"shards introuvables dans {data_dir}. Lance d'abord : python prepare_data.py --out_dir {data_dir}"
        )
    train_shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in train_paths]
    val_shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in val_paths]
    print(f"train: {len(train_shards)} shard(s), {sum(len(s) for s in train_shards):,} tokens")
    print(f"val:   {len(val_shards)} shard(s), {sum(len(s) for s in val_shards):,} tokens")
    return train_shards, val_shards


def get_batch(shards, block_size, batch_size, device):
    xs, ys = [], []
    for _ in range(batch_size):
        shard = shards[np.random.randint(len(shards))]
        i = np.random.randint(len(shard) - block_size)
        xs.append(torch.from_numpy(shard[i:i + block_size].astype(np.int64)))
        ys.append(torch.from_numpy(shard[i + 1:i + 1 + block_size].astype(np.int64)))
    x = torch.stack(xs)
    y = torch.stack(ys)
    if device == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def configure_optimizer(model, weight_decay, lr, betas):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_no_decay = sum(p.numel() for p in no_decay)
    print(f"optimizer: {n_decay:,} params avec weight decay, {n_no_decay:,} sans (normes/biais)")
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


@torch.no_grad()
def estimate_loss(model, data_by_split, block_size, batch_size, device, eval_iters, autocast_ctx):
    model.eval()
    out = {}
    for split, shards in data_by_split.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(shards, block_size, batch_size, device)
            with autocast_ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def print_sample(model, enc, device, max_new_tokens, prompt="The history of"):
    model.eval()
    ids = enc.encode_ordinary(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=0.8, top_k=50)
    text = enc.decode(y[0].tolist())
    model.train()
    print(f"--- echantillon ---\n{text}\n-------------------")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_shards, val_shards = load_data(args.data_dir)
    data_by_split = {"train": train_shards, "val": val_shards}

    use_cuda = args.device == "cuda" and torch.cuda.is_available()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] --device cuda demande mais aucun GPU CUDA detecte, bascule sur cpu")
        args.device = "cpu"

    if use_cuda:
        bf16_ok = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_ok else torch.float16
        autocast_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
        print(f"cuda: autocast dtype={amp_dtype}")
    else:
        import contextlib
        autocast_ctx = contextlib.nullcontext()
        scaler = None
        print("device=cpu : pas d'autocast, float32 (attendu pour un smoke test, pas pour un vrai entrainement)")

    config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        max_seq_len=args.block_size,
        dropout=args.dropout,
    )
    model = ToyLLM(config).to(args.device)
    n_params = model.get_num_params()
    print(f"modele: {n_params['total']:,} params total ({n_params['non_embedding']:,} hors embeddings, "
          f"{n_params['embedding']:,} embeddings)")

    optimizer = configure_optimizer(model, args.weight_decay, args.max_lr, (args.beta1, args.beta2))

    start_step = 0
    best_val_loss = float("inf")
    ckpt_path = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"reprise depuis {ckpt_path} au step {start_step}")

    if args.compile:
        model = torch.compile(model)

    enc = tiktoken.get_encoding("gpt2")

    t0 = time.time()
    tokens_per_step = args.batch_size * args.block_size * args.grad_accum_steps
    print(f"batch effectif: {tokens_per_step:,} tokens/step ({args.batch_size} x {args.block_size} x {args.grad_accum_steps} accum)")
    for step in range(start_step, args.max_steps + 1):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(args.grad_accum_steps):
            x, y = get_batch(train_shards, args.block_size, args.batch_size, args.device)
            with autocast_ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if step % args.log_interval == 0:
            dt = time.time() - t0
            tok_per_sec = tokens_per_step * args.log_interval / max(dt, 1e-9) if step > start_step else 0.0
            print(f"step {step:6d} | loss {loss_accum:.4f} | lr {lr:.2e} | {tok_per_sec:,.0f} tok/s")
            t0 = time.time()

        if step % args.eval_interval == 0 and step > start_step:
            losses = estimate_loss(model, data_by_split, args.block_size, args.batch_size, args.device, args.eval_iters, autocast_ctx)
            print(f"--- eval step {step} : train {losses['train']:.4f} | val {losses['val']:.4f} ---")
            print_sample(model, enc, args.device, args.sample_tokens)
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "config": asdict(config),
                    "best_val_loss": best_val_loss,
                }, os.path.join(args.out_dir, "best.pt"))
                print(f"[best.pt sauvegarde] val_loss={best_val_loss:.4f}")

        if step % args.save_interval == 0 and step > start_step:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
                "config": asdict(config),
                "best_val_loss": best_val_loss,
            }, ckpt_path)
            print(f"[last.pt sauvegarde] step={step}")

    print("Entrainement termine.")


if __name__ == "__main__":
    main()
