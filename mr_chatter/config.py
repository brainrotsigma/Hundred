"""Model configuration for Mr Chatter."""

from dataclasses import dataclass


@dataclass
class MrChatterConfig:
    """Configuration dataclass for the Mr Chatter transformer.

    Attributes:
        vocab_size: Size of the tokenizer vocabulary.
        block_size: Maximum sequence length (context) the model can attend to.
        n_embd: Dimensionality of the token embeddings.
        n_layer: Number of transformer decoder blocks.
        n_head: Number of attention heads per block.
        dropout: Dropout probability applied throughout the model.
    """

    vocab_size: int
    block_size: int = 128
    n_embd: int = 128
    n_layer: int = 4
    n_head: int = 4
    dropout: float = 0.1

    def validate(self) -> None:
        """Validate the configuration values."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head for multi-head attention"
            )
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the [0, 1) interval")
