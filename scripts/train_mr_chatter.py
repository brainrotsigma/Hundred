"""CLI entry point to train and interact with Mr Chatter."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mr_chatter import (
    CharacterTokenizer,
    MrChatter,
    MrChatterConfig,
    MrChatterTrainer,
    TrainingConfig,
    load_corpus,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mr Chatter transformer")
    parser.add_argument("--corpus", type=Path, help="Path to a plain-text corpus", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are Mr Chatter, a curious student eager to learn English.",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="hello friend",
        help="Seed text for generation after training",
    )
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed-value", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Computation device (cpu, cuda, mps, or auto)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed_value)

    corpus = load_corpus(args.corpus)
    tokenizer = CharacterTokenizer.build_from_texts(
        list(corpus) + [args.system_prompt, args.seed]
    )

    config = MrChatterConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )
    model = MrChatter(config)

    trainer_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        log_interval=args.log_interval,
        device=args.device,
    )
    trainer = MrChatterTrainer(model, tokenizer, trainer_config)

    print("Training on", len(corpus), "lines from", args.corpus or "sample_corpus.txt")
    trainer.fit(corpus)

    reply = model.chat(
        tokenizer,
        system_prompt=args.system_prompt,
        user_prompt=args.seed,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print("\nSystem prompt:", args.system_prompt)
    print("User prompt:", args.seed)
    print("Mr Chatter:", reply)


if __name__ == "__main__":
    main()
