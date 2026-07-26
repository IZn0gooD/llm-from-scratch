"""
Phase 1 -- modele Transformer dense from scratch (identique a l'architecture Phase 0, config plus grande).
RoPE (rotate-half) + RMSNorm + SwiGLU + GQA (grouped-query attention),
attention via F.scaled_dot_product_attention (backend flash/efficient auto-selectionne par PyTorch).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 50304      # 50257 (GPT-2 BPE) arrondi a un multiple de 64
    dim: int = 1024
    n_layers: int = 24
    n_heads: int = 16            # tetes de requete
    n_kv_heads: int = 4          # tetes key/value (GQA) ; n_heads % n_kv_heads == 0
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, "dim doit etre divisible par n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads doit etre divisible par n_kv_heads (GQA)"

class RMSNorm(nn.Module):
    """RMSNorm calcule en float32 pour la stabilite numerique, quel que soit le dtype d'entree."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(in_dtype)

def precompute_rope_cache(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    """Retourne cos/sin de forme (max_seq_len, head_dim), convention 'rotate-half' (style Llama/HF)."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, freqs)                       # (max_seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)              # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, Dh) ; cos, sin: (T, Dh) -> broadcast sur B et H."""
    cos = cos[None, None, :, :].to(x.dtype)
    sin = sin[None, None, :, :].to(x.dtype)
    return x * cos + rotate_half(x) * sin

class Attention(nn.Module):
    """Grouped-query attention : n_heads tetes Q partagent n_kv_heads tetes K/V."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.dropout = config.dropout

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # (B, Hq, T, Dh)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, Hkv, T, Dh)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)   # (B, Hq, T, Dh)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class SwiGLU(nn.Module):
    """MLP SwiGLU (Llama-style) : hidden reduit a 2/3 du multiplicateur cible, arrondi a multiple_of."""

    def __init__(self, dim: int, hidden_mult: int = 4, multiple_of: int = 32):
        super().__init__()
        hidden = int(2 * (hidden_mult * dim) / 3)
        hidden = multiple_of * ((hidden + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate
        self.w3 = nn.Linear(dim, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, dim, bias=False)   # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.dim, config.norm_eps)
        self.mlp = SwiGLU(config.dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class ToyLLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm_f = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        head_dim = config.dim // config.n_heads
        cos, sin = precompute_rope_cache(head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

        self.apply(self._init_weights)
        # scaling GPT-2 : attenue les projections qui ecrivent dans le residual stream,
        # pour compenser l'accumulation sur n_layers connexions residuelles
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.tok_emb.weight.numel()
        return {"total": total, "non_embedding": total - emb, "embedding": emb}

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.max_seq_len, f"sequence de longueur {T} > max_seq_len={self.config.max_seq_len}"

        x = self.tok_emb(idx)
        cos = self.cos_cached[:T]
        sin = self.sin_cached[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference : ne calcule les logits que sur le dernier token (economie de calcul)
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
