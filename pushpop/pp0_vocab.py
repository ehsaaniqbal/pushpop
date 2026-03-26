"""Fixed vocabulary for PP0 program and task tokens."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

PAD_TOKEN: Final[str] = "PAD"
OUT_TOKEN: Final[str] = "OUT"
STOP_TOKEN: Final[str] = "STOP"

VOCAB_TOKENS: Final[tuple[str, ...]] = (
    PAD_TOKEN,
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "DUP",
    "POP",
    "SWAP",
    "ADD",
    "SUB",
    "END",
    OUT_TOKEN,
    STOP_TOKEN,
)
TOKEN_TO_ID: Final[dict[str, int]] = {token: index for index, token in enumerate(VOCAB_TOKENS)}
ID_TO_TOKEN: Final[dict[int, str]] = {index: token for token, index in TOKEN_TO_ID.items()}
PAD_ID: Final[int] = TOKEN_TO_ID[PAD_TOKEN]


def encode_tokens(tokens: Sequence[str]) -> list[int]:
    try:
        return [TOKEN_TO_ID[token] for token in tokens]
    except KeyError as error:
        raise ValueError(f"unknown PP0 token: {error.args[0]!r}") from error


def decode_ids(token_ids: Sequence[int]) -> list[str]:
    try:
        return [ID_TO_TOKEN[token_id] for token_id in token_ids]
    except KeyError as error:
        raise ValueError(f"unknown PP0 token id: {error.args[0]!r}") from error
