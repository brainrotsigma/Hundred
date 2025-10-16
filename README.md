# Mr Chatter

Mr Chatter is a compact decoder-only transformer experiment focused on learning the basics of English.
It includes:

- A small GPT-style architecture implemented with PyTorch.
- A minimal character-level tokenizer with support for system prompts.
- Training utilities and a demonstration CLI to fit the model on a tiny English corpus.

The repository is intentionally lightweight so you can explore how decoder-only transformers
work without depending on large external datasets.

## Project structure

```
mr_chatter/
  __init__.py          # Package exports
  config.py            # Model configuration dataclass
  model.py             # Decoder-only transformer implementation
  tokenizer.py         # Character-level tokenizer
  trainer.py           # Training and evaluation helpers
  data.py              # Utilities for loading toy corpora
scripts/
  train_mr_chatter.py  # CLI entry point for training & generation demo
resources/
  sample_corpus.txt    # Tiny default corpus with simple English sentences
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Train the model on the bundled sample corpus and generate a short reply:

```bash
python scripts/train_mr_chatter.py --epochs 50 --system-prompt "You are Mr Chatter, an eager student of the English language." --seed "hello there"
```

You can also provide your own text corpus. The training script accepts plain-text
files and will build a fresh tokenizer that includes the characters present in
the dataset.

```bash
python scripts/train_mr_chatter.py --corpus my_english_notes.txt --epochs 100
```

After training, the script prints a short generation using the supplied system
prompt. You can adjust the sampling temperature, top-k filtering, and the number
of new tokens to generate.

## Notes

- This project is an educational toy. The tiny default dataset is enough to
  verify that the pipeline works, but it will not produce impressive results.
- The implementation is intentionally concise and avoids external helper
  libraries so you can read and understand the full training loop.
- Training on larger corpora or longer sequences may require additional tuning
  (more layers, different optimizer parameters, etc.).
