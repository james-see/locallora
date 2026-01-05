# LocalLoRA Runbook

Complete guide to fine-tuning LLMs on Apple Silicon using LoRA.

## Prerequisites

```bash
# Install dependencies
uv sync

# Verify setup
uv run python -c "import mlx; print('MLX ready')"
```

---

## Step 1: Scrape Research Papers

```bash
# Dry run to preview what will be downloaded
uv run python scrape_gravity_papers.py --dry-run --max-per-query 10

# Download papers (default: 20 per query from arXiv, ADS, Semantic Scholar)
uv run python scrape_gravity_papers.py

# Custom options
uv run python scrape_gravity_papers.py \
    --output-dir papers \
    --max-per-query 30 \
    --sources arxiv semantic_scholar \
    --queries "gravitational waves" "black holes" "quantum gravity"
```

**Output:** PDFs saved to `papers/`

---

## Step 2: Process PDFs to Training Data

```bash
uv run python processgravitydocs.py --pdf-folder papers
```

**Output:**
- `papers_text.jsonl` - Text chunks for training
- `papers_alpaca.jsonl` - Instruction format (optional)

**Estimate training time:**
```bash
uv run python estimate_training.py
```

---

## Step 3: Train LoRA Adapter

### First Training Run

```bash
uv run python train_mlx.py --batch-size 4 --iters 1000
```

**Options:**
| Flag | Default | Description |
|------|---------|-------------|
| `--batch-size` | 4 | Increase to 8 with 128GB RAM |
| `--iters` | 1000 | Training iterations |
| `--num-layers` | 16 | Layers to apply LoRA |
| `--learning-rate` | 1e-5 | Learning rate |
| `--output-dir` | ./output/mlx-lora | Output directory |

**Output:** Adapter saved to `./output/mlx-lora/adapter/`

### Resume/Continue Training

```bash
# Add more papers first (optional)
uv run python scrape_gravity_papers.py --max-per-query 50
uv run python processgravitydocs.py

# Continue from existing adapter
uv run python train_mlx.py --resume auto --iters 500
```

---

## Step 4: Test with MLX (Quick)

```bash
# Interactive chat
uv run python chat_with_model.py

# Single prompt
uv run python -m mlx_lm generate \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --adapter-path ./output/mlx-lora/adapter \
    --prompt "[INST] Explain gravitational waves [/INST]"
```

---

## Step 5: Convert to Ollama

### 5a. Fuse Adapter into Base Model

```bash
uv run python -m mlx_lm fuse \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --adapter-path ./output/mlx-lora/adapter \
    --save-path ./output/gravity-expert-8b
```

### 5b. Convert to GGUF Format

```bash
uv run python /Users/jc/projects/llama-cpp-convert/convert_hf_to_gguf.py \
    ./output/gravity-expert-8b \
    --outfile ./output/gravity-expert-8b.gguf \
    --outtype f16
```

**Quantization options:**
- `f16` - Full precision (15GB)
- `q8_0` - 8-bit quantized (~8GB)
- `q4_0` - 4-bit quantized (~4GB)

### 5c. Create Ollama Model

```bash
cd ./output

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./gravity-expert-8b.gguf

TEMPLATE """[INST] {{ .Prompt }} [/INST]"""

PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"
EOF

# Import to Ollama
ollama create gravity-expert -f Modelfile
```

### 5d. Run with Ollama

```bash
ollama run gravity-expert
```

---

## Quick Reference

### Full Pipeline (Copy-Paste)

```bash
# 1. Scrape
uv run python scrape_gravity_papers.py --max-per-query 20

# 2. Process
uv run python processgravitydocs.py

# 3. Train
uv run python train_mlx.py --batch-size 4 --iters 1000

# 4. Fuse
uv run python -m mlx_lm fuse \
    --model mistralai/Ministral-8B-Instruct-2410 \
    --adapter-path ./output/mlx-lora/adapter \
    --save-path ./output/gravity-expert-8b

# 5. Convert
uv run python /Users/jc/projects/llama-cpp-convert/convert_hf_to_gguf.py \
    ./output/gravity-expert-8b \
    --outfile ./output/gravity-expert-8b.gguf \
    --outtype f16

# 6. Create Ollama model
cd output
echo 'FROM ./gravity-expert-8b.gguf
TEMPLATE """[INST] {{ .Prompt }} [/INST]"""
PARAMETER stop "[INST]"
PARAMETER stop "[/INST]"' > Modelfile
ollama create gravity-expert -f Modelfile

# 7. Run
ollama run gravity-expert
```

### Update Existing Model

```bash
# Add more training data
uv run python scrape_gravity_papers.py --max-per-query 50
uv run python processgravitydocs.py

# Continue training
uv run python train_mlx.py --resume auto --iters 500

# Re-export to Ollama (repeat steps 4-6)
```

---

## Troubleshooting

**Out of memory:**
- Reduce `--batch-size` to 2 or 1
- Reduce `--num-layers` to 8

**Training too slow:**
- Increase `--batch-size` (if RAM allows)
- Use MLX instead of PyTorch (`train_mlx.py` vs `train_lora.py`)

**Ollama model not responding well:**
- Train for more iterations
- Add more diverse training data
- Try lower learning rate (`--learning-rate 5e-6`)

---

## File Structure

```
locallora/
├── papers/                    # Downloaded PDFs
├── papers_text.jsonl          # Processed training data
├── config.yml                 # Training config
├── output/
│   ├── mlx-lora/
│   │   ├── adapter/           # LoRA weights
│   │   └── data/              # Processed MLX format
│   ├── gravity-expert-8b/     # Fused model
│   └── gravity-expert-8b.gguf # Ollama-ready model
├── scrape_gravity_papers.py   # Paper scraper
├── processgravitydocs.py      # PDF processor
├── train_mlx.py               # MLX trainer
├── chat_with_model.py         # Test interface
└── estimate_training.py       # Time estimator
```
