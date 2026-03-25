"""Reference interpreter for Pushpop-0 (PP0).

Execution semantics, restated:

- Programs are linear postfix token sequences ending with ``END``.
- Tokens are whitespace-separated when provided as a string.
- Digits ``0``..``9`` push themselves onto the stack.
- ``DUP`` duplicates the top item, ``POP`` removes it, and ``SWAP`` swaps the top
  two items.
- ``ADD`` and ``SUB`` pop the top two items ``a b`` and push a single result in
  ``Z_10``. ``SUB`` is order-sensitive and computes ``(a - b) mod 10``.
- The interpreter is strict about malformed input and raises
  ``PP0ExecutionError``. Data generation should still enforce validity upstream.

The trace captures one record per executed token, including ``END``, with the
full stack before and after that step. This makes the trace easy to align with
token positions later during probing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

MODULUS: Final[int] = 10
HALT_TOKEN: Final[str] = "END"
STRUCTURAL_TOKENS: Final[frozenset[str]] = frozenset({"DUP", "POP", "SWAP"})
ARITHMETIC_TOKENS: Final[frozenset[str]] = frozenset({"ADD", "SUB"})
LITERAL_VALUES: Final[dict[str, int]] = {str(i): i for i in range(MODULUS)}
VALID_TOKENS: Final[frozenset[str]] = frozenset(
    {*LITERAL_VALUES, *STRUCTURAL_TOKENS, *ARITHMETIC_TOKENS, HALT_TOKEN}
)

type Stack = tuple[int, ...]
type ProgramInput = str | Sequence[str]


class PP0ExecutionError(ValueError):
    """Raised when the interpreter receives malformed or invalid input."""


@dataclass(frozen=True, slots=True)
class TraceStep:
    """One executed PP0 token plus machine state before and after it."""

    step: int
    pc: int
    token: str
    stack_before: Stack
    stack_after: Stack
    top_before: int | None
    top_after: int | None

    def as_dict(self) -> dict[str, object]:
        """Return a plain row-shaped record for analysis code."""

        return {
            "step": self.step,
            "pc": self.pc,
            "token": self.token,
            "stack_before": self.stack_before,
            "stack_after": self.stack_after,
            "depth_before": len(self.stack_before),
            "depth_after": len(self.stack_after),
            "top_before": self.top_before,
            "top_after": self.top_after,
        }


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Trace plus final machine state for one PP0 program."""

    tokens: tuple[str, ...]
    trace: tuple[TraceStep, ...]
    final_stack: Stack
    final_top: int | None

    def trace_rows(self) -> list[dict[str, object]]:
        """Return the execution trace as plain Python rows."""

        return [step.as_dict() for step in self.trace]


def tokenize(program: ProgramInput) -> tuple[str, ...]:
    """Normalize a PP0 program into a tuple of tokens."""

    if isinstance(program, str):
        tokens = tuple(program.split())
    else:
        tokens = tuple(program)

    if not tokens:
        raise PP0ExecutionError("program must contain at least one instruction and END")

    for token in tokens:
        if not isinstance(token, str):
            raise TypeError("program tokens must be strings")

    return tokens


def execute(program: ProgramInput) -> ExecutionResult:
    """Execute a PP0 program and return its full execution trace."""

    tokens = tokenize(program)
    _validate_program_shape(tokens)

    stack: list[int] = []
    trace: list[TraceStep] = []

    for pc, token in enumerate(tokens):
        stack_before = tuple(stack)

        if token in LITERAL_VALUES:
            stack.append(LITERAL_VALUES[token])
        elif token == "DUP":
            _require_depth(stack, required=1, token=token, pc=pc)
            stack.append(stack[-1])
        elif token == "POP":
            _require_depth(stack, required=1, token=token, pc=pc)
            stack.pop()
        elif token == "SWAP":
            _require_depth(stack, required=2, token=token, pc=pc)
            stack[-2], stack[-1] = stack[-1], stack[-2]
        elif token == "ADD":
            _require_depth(stack, required=2, token=token, pc=pc)
            right = stack.pop()
            left = stack.pop()
            stack.append((left + right) % MODULUS)
        elif token == "SUB":
            _require_depth(stack, required=2, token=token, pc=pc)
            right = stack.pop()
            left = stack.pop()
            stack.append((left - right) % MODULUS)
        elif token == HALT_TOKEN:
            pass
        else:
            raise PP0ExecutionError(f"unknown token {token!r} at pc {pc}")

        stack_after = tuple(stack)
        trace.append(
            TraceStep(
                step=len(trace),
                pc=pc,
                token=token,
                stack_before=stack_before,
                stack_after=stack_after,
                top_before=stack_before[-1] if stack_before else None,
                top_after=stack_after[-1] if stack_after else None,
            )
        )

    final_stack = tuple(stack)
    return ExecutionResult(
        tokens=tokens,
        trace=tuple(trace),
        final_stack=final_stack,
        final_top=final_stack[-1] if final_stack else None,
    )


def _validate_program_shape(tokens: tuple[str, ...]) -> None:
    if len(tokens) < 2:
        raise PP0ExecutionError("program must contain at least one instruction before END")

    for pc, token in enumerate(tokens):
        if token not in VALID_TOKENS:
            raise PP0ExecutionError(f"unknown token {token!r} at pc {pc}")

    if HALT_TOKEN in tokens[:-1]:
        raise PP0ExecutionError("END may only appear as the final token")

    if tokens[-1] != HALT_TOKEN:
        raise PP0ExecutionError("program must end with END")


def _require_depth(stack: list[int], required: int, token: str, pc: int) -> None:
    if len(stack) < required:
        raise PP0ExecutionError(
            f"{token} requires stack depth >= {required} at pc {pc}, got {len(stack)}"
        )
