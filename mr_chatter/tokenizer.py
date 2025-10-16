"""A lightweight character-level tokenizer for Mr Chatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


_SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<system>",
    "<user>",
    "<assistant>",
]


@dataclass
class TokenizerState:
    stoi: dict[str, int]
    itos: List[str]


class CharacterTokenizer:
    """A simple character-level tokenizer with special prompt tokens."""

    pad_token = "<pad>"
    bos_token = "<bos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    system_token = "<system>"
    user_token = "<user>"
    assistant_token = "<assistant>"

    def __init__(self, state: TokenizerState) -> None:
        self._state = state

    @classmethod
    def build_from_texts(cls, texts: Iterable[str]) -> "CharacterTokenizer":
        """Create a tokenizer from an iterable of texts."""
        charset = set()
        for text in texts:
            charset.update(text)
        # Ensure deterministic order by sorting.
        ordered_chars = sorted(charset)
        itos: List[str] = list(_SPECIAL_TOKENS) + ordered_chars
        stoi = {token: idx for idx, token in enumerate(itos)}
        return cls(TokenizerState(stoi=stoi, itos=itos))

    @property
    def vocab_size(self) -> int:
        return len(self._state.itos)

    def token_to_id(self, token: str) -> int:
        return self._state.stoi.get(token, self._state.stoi[self.unk_token])

    def id_to_token(self, idx: int) -> str:
        return self._state.itos[idx]

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = True,
        role: str | None = None,
    ) -> List[int]:
        """Encode a string into token ids.

        Args:
            text: Raw text to encode.
            add_bos: Whether to prepend the beginning-of-sequence token.
            add_eos: Whether to append the end-of-sequence token.
            role: Optional dialogue role ("system", "user", "assistant").
        """
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.token_to_id(self.bos_token))
        if role is not None:
            role_token = {
                "system": self.system_token,
                "user": self.user_token,
                "assistant": self.assistant_token,
            }.get(role.lower())
            if role_token is None:
                raise ValueError(f"Unsupported role: {role}")
            tokens.append(self.token_to_id(role_token))
        for ch in text:
            tokens.append(self.token_to_id(ch))
        if add_eos:
            tokens.append(self.token_to_id(self.eos_token))
        return tokens

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a list of token ids back into a string."""
        pieces: List[str] = []
        for idx in token_ids:
            token = self.id_to_token(int(idx))
            if token == self.eos_token:
                break
            if token in _SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces)

    def format_dialogue(
        self,
        *,
        system_prompt: str | None,
        user_prompt: str,
    ) -> List[int]:
        """Format a dialogue turn as token ids.

        The returned sequence begins with a BOS token, optionally includes a
        system prompt, the user prompt, and finishes with the <assistant>
        token so that the model can continue the dialogue.
        """
        tokens = [self.token_to_id(self.bos_token)]
        if system_prompt:
            tokens.extend(
                self.encode(system_prompt, add_bos=False, add_eos=True, role="system")
            )
        tokens.extend(self.encode(user_prompt, add_bos=False, add_eos=True, role="user"))
        tokens.append(self.token_to_id(self.assistant_token))
        return tokens
