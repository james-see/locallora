#!/usr/bin/env python3
"""
Chat with your fine-tuned LoRA model using MLX.
"""

import argparse
from mlx_lm import load, generate


def main():
    parser = argparse.ArgumentParser(description="Chat with fine-tuned model")
    parser.add_argument(
        "--model",
        default="mistralai/Ministral-8B-Instruct-2410",
        help="Base model",
    )
    parser.add_argument(
        "--adapter",
        default="./output/mlx-lora/adapter",
        help="Path to LoRA adapter",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"Loading adapter: {args.adapter}")

    # Load model with adapter
    model, tokenizer = load(args.model, adapter_path=args.adapter)

    print("\n" + "=" * 60)
    print("Model loaded! Type your questions (Ctrl+C to exit)")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("You: ").strip()
            if not prompt:
                continue

            # Format as chat (Mistral instruct format)
            formatted = f"[INST] {prompt} [/INST]"

            response = generate(
                model,
                tokenizer,
                prompt=formatted,
                max_tokens=args.max_tokens,
            )

            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
