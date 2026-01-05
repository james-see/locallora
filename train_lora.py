#!/usr/bin/env python3
"""
LoRA fine-tuning script for Apple Silicon MPS.
Uses PEFT + TRL instead of Axolotl for better macOS compatibility.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_dataset(jsonl_path: str, text_field: str = "text") -> Dataset:
    """Load JSONL dataset."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                data.append({text_field: item.get(text_field, "")})
    return Dataset.from_list(data)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize text examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on MPS")
    parser.add_argument("config", help="Path to config YAML file")
    parser.add_argument("--dataset", default="papers_text.jsonl", help="JSONL dataset path")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA")
    else:
        device = "cpu"
        print("Using CPU")

    # Load tokenizer
    print(f"Loading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        use_fast=config.get("tokenizer_use_fast", True),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        torch_dtype=torch.bfloat16 if config.get("bf16", False) else torch.float32,
        device_map="auto" if device != "mps" else None,
        trust_remote_code=True,
    )

    if device == "mps":
        model = model.to(device)

    # Configure LoRA
    print("Configuring LoRA adapter")
    lora_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable input gradients for gradient checkpointing compatibility
    model.enable_input_require_grads()

    # Load and tokenize dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset)
    max_length = config.get("sequence_len", 2048)

    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    output_dir = config.get("output_dir", "./output/lora")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("micro_batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 16),
        learning_rate=config.get("learning_rate", 2e-5),
        weight_decay=config.get("weight_decay", 0.0),
        warmup_steps=config.get("warmup_steps", 100),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        max_grad_norm=config.get("max_grad_norm", 1.0),
        logging_steps=config.get("logging_steps", 50),
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 5),
        bf16=config.get("bf16", False) and device != "mps",  # bf16 can be tricky on MPS
        fp16=False,
        gradient_checkpointing=config.get("gradient_checkpointing", True) and device != "mps",
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
        dataloader_pin_memory=False if device == "mps" else True,
        use_cpu=device == "cpu",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save adapter
    adapter_path = Path(output_dir) / "adapter"
    print(f"Saving adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print("Training complete!")


if __name__ == "__main__":
    main()
