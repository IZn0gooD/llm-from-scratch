# LLM from scratch

Projet perso : construire un LLM from scratch en PyTorch pur, avec pour objectif le controle total sur l'architecture et les meilleurs resultats possibles meme a petite echelle (curation des donnees > scaling brut).

## Phase 0 -- modele jouet (`phase0/`)

Transformer dense ~28M parametres, from scratch, sans framework haut niveau (pas de `Trainer`, pas d'abstraction cachee) :

- **RoPE** (rotary position embeddings, convention rotate-half style Llama/HF)
- **RMSNorm**
- **SwiGLU** (MLP style Llama)
- **GQA** (grouped-query attention) via `F.scaled_dot_product_attention` (backend flash/efficient auto-selectionne par PyTorch)
- Entrainement sur [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (tokenizer GPT-2 / `tiktoken`)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# torch : choisir la bonne build selon le hardware
pip install torch --index-url https://download.pytorch.org/whl/cpu   # pas de GPU NVIDIA
pip install torch                                                    # GPU cuda (cloud, Colab, Kaggle)

pip install -r phase0/requirements.txt
```

### Usage

```bash
# 1. Telecharger + tokeniser TinyStories (~2.2 Go, quelques minutes)
python phase0/prepare_data.py

# 2. Entrainer (config par defaut : ~28M params, dim=384/6 layers/6 heads/2 kv_heads)
python phase0/train.py

# reprise apres interruption (utile sur GPU gratuit a session limitee : Colab/Kaggle)
python phase0/train.py --resume

# 3. Generer du texte depuis un checkpoint
python phase0/sample.py --ckpt phase0/checkpoints/best.pt --prompt "Once upon a time,"
```

### Test rapide (sans tout telecharger)

```bash
python phase0/prepare_data.py --out_dir phase0/data_smoke --max_download_mb 5 --limit_train_stories 300 --limit_val_stories 100
python phase0/train.py --data_dir phase0/data_smoke --out_dir phase0/checkpoints_smoke \
  --dim 64 --n_layers 2 --n_heads 2 --n_kv_heads 1 --block_size 64 --batch_size 8 --max_steps 30
```

Toutes les options d'entrainement (batch size, LR, warmup, intervalles d'eval/sauvegarde...) sont listables via `python phase0/train.py --help`.
