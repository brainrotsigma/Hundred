"""Training utilities for Mr Chatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .model import MrChatter
from .tokenizer import CharacterTokenizer


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.95)
    device: str = "auto"
    log_interval: int = 25


class _SlidingWindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Produce (input, target) pairs from a continuous token stream."""

    def __init__(
        self,
        encoded_stream: torch.Tensor,
        block_size: int,
    ) -> None:
        super().__init__()
        if encoded_stream.ndim != 1:
            raise ValueError("encoded_stream must be 1D")
        if len(encoded_stream) <= block_size:
            raise ValueError("encoded_stream must be longer than block_size")
        self.data = encoded_stream
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size]
        target = self.data[idx + 1 : idx + self.block_size + 1]
        return chunk, target


class MrChatterTrainer:
    """Simple trainer for the Mr Chatter model."""

    def __init__(
        self,
        model: MrChatter,
        tokenizer: CharacterTokenizer,
        config: TrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.device = self._resolve_device(self.config.device)
        self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _build_dataset(self, texts: Iterable[str], block_size: int) -> _SlidingWindowDataset:
        token_ids = []
        for text in texts:
            token_ids.extend(
                self.tokenizer.encode(text, add_bos=True, add_eos=True, role="assistant")
            )
        encoded = torch.tensor(token_ids, dtype=torch.long)
        if len(encoded) < 2:
            raise ValueError("Corpus is too small to create training samples")
        if len(encoded) <= block_size:
            block_size = max(1, len(encoded) - 1)
        return _SlidingWindowDataset(encoded_stream=encoded, block_size=block_size)

    def fit(self, texts: Iterable[str], *, block_size: int | None = None) -> None:
        """Train the model on the provided texts."""
        cfg = self.config
        block = block_size or self.model.config.block_size
        dataset = self._build_dataset(texts, block)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        step = 0
        for epoch in range(cfg.epochs):
            running_loss = 0.0
            for batch_idx, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                logits, loss = self.model(x, targets=y)
                if loss is None:
                    # Fall back to manual loss computation if model did not return it.
                    vocab_size = logits.size(-1)
                    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                step += 1

                if cfg.log_interval and step % cfg.log_interval == 0:
                    avg_loss = running_loss / cfg.log_interval
                    print(f"epoch {epoch+1} step {step}: loss={avg_loss:.4f}")
                    running_loss = 0.0

            if cfg.log_interval and running_loss:
                avg_loss = running_loss / (len(loader) % cfg.log_interval or cfg.log_interval)
                print(f"epoch {epoch+1} end: loss={avg_loss:.4f}")

        print("Training complete.")
