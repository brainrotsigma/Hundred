"""Decoder-only transformer implementation for Mr Chatter."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .config import MrChatterConfig
from .tokenizer import CharacterTokenizer


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""

    def __init__(self, config: MrChatterConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # Register a causal mask to avoid using future tokens.
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("attn_mask", mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = self.attn_mask[:, :, :T, :T]
        att = att.masked_fill(mask == 0, torch.finfo(att.dtype).min)
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class MLP(nn.Module):
    """Transformer feed-forward block."""

    def __init__(self, config: MrChatterConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = F.gelu(x)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block composed of self-attention and feed-forward."""

    def __init__(self, config: MrChatterConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MrChatter(nn.Module):
    """Decoder-only transformer backbone for Mr Chatter."""

    def __init__(self, config: MrChatterConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        # Tie weights for efficiency.
        self.lm_head.weight = self.token_embedding.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        device = idx.device
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block size")

        positions = torch.arange(0, T, device=device, dtype=torch.long)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)[None, :, :]
        x = self.drop(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        *,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        generated = idx
        for _ in range(max_new_tokens):
            idx_cond = generated[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_values = values[:, -1].unsqueeze(-1)
                cutoff = torch.full_like(logits, torch.finfo(logits.dtype).min)
                logits = torch.where(logits < min_values, cutoff, logits)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

    @torch.no_grad()
    def chat(
        self,
        tokenizer: CharacterTokenizer,
        *,
        system_prompt: str | None,
        user_prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: Optional[int] = 20,
    ) -> str:
        """Generate a reply using an optional system prompt and user prompt."""
        device = next(self.parameters()).device
        context_tokens = tokenizer.format_dialogue(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        context = torch.tensor([context_tokens], dtype=torch.long, device=device)
        generated = self.generate(
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # Decode the generated portion after the context
        new_tokens = generated[0, context.size(1) :].tolist()
        return tokenizer.decode(new_tokens)
