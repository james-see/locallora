#!/usr/bin/env python3
"""Estimate LoRA training time based on dataset and config."""

import argparse
import json
from pathlib import Path

import yaml


def count_tokens_approx(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(description="Estimate LoRA training time")
    parser.add_argument("--dataset", default="papers_text.jsonl", help="JSONL dataset file")
    parser.add_argument("--config", default="config.yml", help="Axolotl config file")
    parser.add_argument(
        "--sec-per-step",
        type=float,
        default=2.5,
        help="Seconds per training step (M4 Max ~2-3s, adjust based on your hardware)",
    )
    args = parser.parse_args()

    # Load config
    config = yaml.safe_load(Path(args.config).read_text())
    seq_len = config.get("sequence_len", 2048)
    epochs = config.get("num_epochs", 3)
    micro_batch = config.get("micro_batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 16)
    effective_batch = micro_batch * grad_accum

    # Count samples and tokens
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        return

    total_tokens = 0
    num_samples = 0
    total_chars = 0

    with dataset_path.open() as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = item.get("text", "")
                total_chars += len(text)
                total_tokens += count_tokens_approx(text)
                num_samples += 1

    # Calculate training sequences (samples may be chunked/packed by seq_len)
    avg_tokens_per_sample = total_tokens // max(num_samples, 1)
    sequences_per_sample = max(1, avg_tokens_per_sample // seq_len)
    total_sequences = num_samples * sequences_per_sample

    # Training steps
    steps_per_epoch = total_sequences // effective_batch
    total_steps = steps_per_epoch * epochs

    # Time estimate
    total_seconds = total_steps * args.sec_per_step
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    print("=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"  Samples:           {num_samples:,}")
    print(f"  Total characters:  {total_chars:,}")
    print(f"  Est. tokens:       {total_tokens:,}")
    print(f"  Avg tokens/sample: {avg_tokens_per_sample:,}")
    print()
    print("=" * 50)
    print("Training Configuration")
    print("=" * 50)
    print(f"  Sequence length:   {seq_len}")
    print(f"  Epochs:            {epochs}")
    print(f"  Micro batch size:  {micro_batch}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch:   {effective_batch}")
    print()
    print("=" * 50)
    print("Training Estimate")
    print("=" * 50)
    print(f"  Total sequences:   {total_sequences:,}")
    print(f"  Steps per epoch:   {steps_per_epoch:,}")
    print(f"  Total steps:       {total_steps:,}")
    print(f"  Sec per step:      {args.sec_per_step}")
    print()
    print(f"  Estimated time:    {int(hours)}h {int(minutes)}m")
    print("=" * 50)
    print()
    print("Note: Actual time varies with hardware, model size, and MPS efficiency.")
    print("      Run a few steps first to calibrate --sec-per-step for your setup.")


if __name__ == "__main__":
    main()
