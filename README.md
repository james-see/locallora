# LocalLoRA

LoRA fine-tuning on Apple Silicon M4 Max using gravity research papers.

## Setup

```bash
# Install dependencies with uv
uv sync

# Verify MPS support
uv run python -c "import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"
```

### Optional: NASA ADS API Key

For NASA ADS paper scraping, get a free API key from https://ui.adsabs.harvard.edu/user/settings/token and set:

```bash
export ADS_DEV_KEY="your-key-here"
```

## Workflow

### 1. Scrape Gravity Papers

```bash
# Default: search all sources for built-in gravity queries
uv run python scrape_gravity_papers.py

# Dry run to see what would be downloaded
uv run python scrape_gravity_papers.py --dry-run

# Custom options
uv run python scrape_gravity_papers.py \
    --output-dir papers \
    --max-per-query 30 \
    --sources arxiv semantic_scholar \
    --queries "gravitational waves" "black holes"
```

### 2. Process PDFs to JSONL

```bash
uv run python processgravitydocs.py --pdf-folder papers
```

Outputs:
- `papers_text.jsonl` - chunked text for training
- `papers_alpaca.jsonl` - Alpaca-style format for supervised fine-tuning

### 3. Run LoRA Training

**Option A: MLX (Recommended for Apple Silicon - 5-10x faster)**

```bash
uv run python train_mlx.py --batch-size 4 --iters 1000
```

**Option B: PyTorch MPS (slower, but more compatible)**

```bash
uv run python train_lora.py config.yml
```

Output adapter saved to `./output/mlx-lora/adapter` or `./output/lora-m4max/adapter`

## Configuration

Edit `config.yml` to adjust:
- `micro_batch_size` - increase if memory allows (1-2 for M4 Max)
- `gradient_accumulation_steps` - effective batch = micro_batch × accumulation
- `num_epochs` - training iterations
- `sequence_len` - max token length per sample

See `Setup-Notes-Config.md` for detailed parameter explanations.