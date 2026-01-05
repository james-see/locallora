#!/usr/bin/env python3
"""
MLX-based LoRA fine-tuning for Apple Silicon.
Much faster than PyTorch MPS.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def convert_jsonl_to_mlx_format(input_path: str, output_dir: str) -> tuple[str, str]:
    """
    Convert papers_text.jsonl to MLX training format.
    MLX expects {"text": "..."} format, one per line.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"

    data = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # MLX expects just {"text": "..."}
                data.append({"text": item.get("text", "")})

    # Split 95/5 train/valid
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    valid_data = data[split_idx:]

    with open(train_file, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(valid_file, "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")

    print(f"Created {len(train_data)} training samples, {len(valid_data)} validation samples")
    return str(train_file), str(valid_file)


def main():
    parser = argparse.ArgumentParser(description="MLX LoRA fine-tuning for Apple Silicon")
    parser.add_argument("--config", default="config.yml", help="Config file path")
    parser.add_argument("--dataset", default="papers_text.jsonl", help="Input JSONL dataset")
    parser.add_argument("--output-dir", default="./output/mlx-lora", help="Output directory")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (can be higher on M4 Max)"
    )
    parser.add_argument("--iters", type=int, default=1000, help="Training iterations")
    parser.add_argument("--num-layers", type=int, default=16, help="Number of layers to apply LoRA")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--steps-per-report", type=int, default=10, help="Report every N steps")
    parser.add_argument("--steps-per-eval", type=int, default=100, help="Eval every N steps")
    parser.add_argument("--save-every", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from existing adapter (path to adapters.safetensors)",
    )
    args = parser.parse_args()

    # Load config for model name
    config = yaml.safe_load(Path(args.config).read_text())
    model_name = config.get("base_model", "mistralai/Ministral-8B-Instruct-2410")

    # Prepare data directory
    data_dir = Path(args.output_dir) / "data"
    train_file, valid_file = convert_jsonl_to_mlx_format(args.dataset, str(data_dir))

    # Build MLX command (using new mlx_lm CLI format)
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model_name,
        "--data",
        str(data_dir),
        "--train",
        "--batch-size",
        str(args.batch_size),
        "--num-layers",
        str(args.num_layers),
        "--iters",
        str(args.iters),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--save-every",
        str(args.save_every),
        "--adapter-path",
        str(Path(args.output_dir) / "adapter"),
    ]

    # Add resume flag if provided
    if args.resume:
        if args.resume == "auto":
            resume_path = Path(args.output_dir) / "adapter" / "adapters.safetensors"
        else:
            resume_path = Path(args.resume)

        if not resume_path.exists():
            print(f"Warning: Resume adapter not found at {resume_path}")
            print("Starting fresh training instead.")
            resume_path = None

        if resume_path:
            cmd.extend(["--resume-adapter-file", str(resume_path)])
            print(f"Resuming from: {resume_path}")

    print("\n" + "=" * 60)
    print("Starting MLX LoRA Training" + (" (RESUMING)" if args.resume else ""))
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch size: {args.batch_size}")
    print(f"Iterations: {args.iters}")
    print(f"Output: {args.output_dir}")
    if args.resume:
        print(f"Resume from: {args.resume}")
    print("=" * 60 + "\n")

    # Run training
    subprocess.run(cmd, check=True)

    print("\n" + "=" * 60)
    print(f"Training complete! Adapter saved to: {Path(args.output_dir) / 'adapter'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
