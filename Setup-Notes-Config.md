base_model: meta-llama/Llama-2-7b-hf # change if using a different model; prefer smaller/optimized model for M4
tokenizer:
use_fast: true

datasets:

path: ./papers_text.jsonl
type: text
max_samples: -1
shuffle: true
LoRA adapter training (PEFT)
lora:
r: 8
alpha: 32
dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
bias: none

training:
epochs: 3
micro_batch_size: 1 # per-device micro-batch; keep small for MPS
grad_accumulation_steps: 16 # accumulate to simulate larger batch
learning_rate: 2e-5
weight_decay: 0.0
warmup_steps: 100
lr_scheduler: cosine
optimizer: adamw_torch
max_grad_norm: 1.0

precision:
fp16: false # MPS fp16 support is limited — test locally; set to true only if your PyTorch/MPS stack supports it safely
bf16: true # bf16 may be supported on Apple Silicon Mx chips—test and toggle if unsupported

checkpointing:
save_strategy: steps
save_steps: 500
save_total_limit: 5
output_dir: ./output/lora-m4max

logging:
report_to: tensorboard
logging_steps: 50
eval_steps: 500

evaluation:
do_eval: false # set true and provide eval dataset if desired
eval_dataset:
path: ./papers_eval.jsonl
type: text
eval_max_samples: 200

distributed:
deepspeed: false
ddp: false

tokenizer_and_data:
max_seq_length: 2048
pad_to_max_length: false
trim_long_sequences: true
chunking:
enabled: true
chunk_size: 1024
chunk_overlap: 128

training_utils:
gradient_checkpointing: true
use_cache: false
gradient_checkpointing_reduce_cpu_memory: true

hardware:
use_gpu: true # instructs Axolotl to use available accelerator
device: mps # specify 'mps' for Apple Silicon backend; if Axolotl doesn't accept this, set device_map: auto in runtime args

save:
final_adapter_path: ./output/lora-m4max/adapter

notes:

Set micro_batch_size and grad_accumulation_steps to fit GPU memory. On M4 Max you can experiment with micro_batch_size=1..2 and accumulate more steps since you have large unified RAM.
If you see errors related to bf16/float16, toggle precision.bf16 and precision.fp16 accordingly.
When using models from Hugging Face, prefer models converted to the HF format and compatible with CPU/MPS. Very large models (30B+) are unlikely to run effectively on MPS.
Test the pipeline on a small subset first (--max_samples: 50) to confirm everything works.
Why QLoRA isn’t a local option on M4 (and recommended alternatives)
QLoRA uses bitsandbytes for 4-bit quantization and training. bitsandbytes depends on CUDA and is not available for Apple Silicon MPS. Therefore:
You cannot run QLoRA natively on your Mac M4.
Option A (recommended if you want QLoRA): run QLoRA on a cloud instance (AWS, GCP, Lambda Labs, Paperspace) or a remote NVIDIA workstation, then convert results (adapter or weights) for local use.
Option B: run standard LoRA/PEFT on M4 with small batches and gradient checkpointing (what the config above targets).
Option C: use llama.cpp / ggml-metal and do inference & small adapter tuning approaches supported by that ecosystem (some adapter workflows exist for GGUF / ggml). This is inference-first, not full LoRA fine-tuning in PyTorch.
Mac-specific setup tips
Install PyTorch with MPS support from the official PyTorch instructions for macOS (they have macOS wheels supporting MPS). Then confirm:
python -c "import torch; print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"

Install Axolotl + deps:
pip install axolotl[all] transformers datasets accelerate peft

If Axolotl or PEFT attempts to call bitsandbytes (for QLoRA), ensure configs do not enable QLoRA. Use the LoRA block shown above.

Monitor resource usage:

On macOS, Activity Monitor + top can show memory usage. Use small test runs to dial micro_batch_size and grad_accumulation_steps.
Suggested workflow (practical step-by-step)
Step 1: Convert PDFs
Run the pdfs_to_jsonl.py script to create papers_text.jsonl and papers_alpaca.jsonl.
Step 2: Test locally (quick run)
Create a tiny dataset papers_test.jsonl with 10 chunks.
Run Axolotl training on that to validate pipeline and device configuration.
Step 3: Choose supervised vs unsupervised
If you have human-written summaries/labels, populate the output fields in the Alpaca JSONL.
If not, consider generating synthetic summaries using a local Ollama model or a cloud LLM, then review.
Step 4: Full LoRA training on M4
Use the provided your_config.yml, tweak micro_batch_size and precision after test runs.
Step 5 (optional): For QLoRA / heavy experiments
Move to a CUDA GPU instance; run QLoRA there; if you get adapter files, you can keep them and use Ollama if it supports adapters, or convert final model to GGUF for local inference with llama.cpp (Metal-enabled) for best local speed.