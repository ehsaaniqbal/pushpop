# Pushpop-0 (PP0)

`PP0` is a tiny linear postfix stack language for the first `pushpop` experiments.

Design goals:

- Train quickly on small models
- Force real hidden state tracking
- Keep execution and evaluation trivial
- Stay legible enough to inspect by hand

This version has no control flow, no memory beyond the stack, and no invalid programs in the training set.

## Summary

- Program tokens: `0..9`, `DUP`, `POP`, `SWAP`, `ADD`, `SUB`, `END`
- Values live in `Z_10` (arithmetic is modulo 10)
- Max stack depth: `4`
- Final stack depth: `1..4`
- Recommended first task: full final stack prediction

Depth `1..4` with values `0..9` already gives `11,110` possible final stacks, which is enough state diversity for a first pass without making the setup messy.

## Instruction Set

- `0..9`: push that literal onto the stack
- `DUP`: duplicate the top item
- `POP`: remove the top item
- `SWAP`: swap the top two items
- `ADD`: pop top two `a b`, push `(a + b) mod 10`
- `SUB`: pop top two `a b`, push `(a - b) mod 10`
- `END`: halt

Deliberate omissions for `v0`:

- No `MUL`
- No `ROT`
- No variables or memory cells
- No jumps, branches, or loops

## Grammar

```text
program := instr+ END
instr   := digit | DUP | POP | SWAP | ADD | SUB
digit   := 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
```

Tokens are whitespace-separated. Digits are instructions, not arguments to a separate `PUSH` opcode.

## Execution Semantics

- Execute left to right.
- Stack notation is `[bottom, ..., top]`.
- All values are integers modulo 10.
- `DUP` and `POP` require stack depth `>= 1`.
- `SWAP`, `ADD`, and `SUB` require stack depth `>= 2`.
- Effects on stack depth:
  - digit, `DUP`: `+1`
  - `POP`, `ADD`, `SUB`: `-1`
  - `SWAP`: `0`
- `SUB` is postfix order:
  - if the stack ends with `[..., x, y]`
  - after `SUB` it becomes `[..., (x - y) mod 10]`

For `v0`, semantics are undefined on invalid prefixes. The data generator should only emit valid programs.

## Program Generation Constraints

Training programs should satisfy:

- `6` to `18` instructions before `END`
- Max stack depth `4`
- Final stack depth `1..4`
- Every prefix must be executable
- At least one arithmetic op: `ADD` or `SUB`
- At least one structural op: `DUP`, `POP`, or `SWAP`
- No more than `3` consecutive literal pushes

Generation rule:

- Sample left to right while tracking current stack depth
- Only allow tokens whose preconditions hold
- Only allow tokens that still leave some path to the target final depth with the remaining steps

Optional cleanup:

- Reject immediate trivial cancellations like `DUP POP` and `SWAP SWAP`

Do not spend time on aggressive equivalence-class deduplication in `v0`.

## Example Programs

All arithmetic below is modulo 10. Final stacks are shown as `[bottom, ..., top]`.

| Program | Final stack |
| --- | --- |
| `2 3 ADD END` | `[5]` |
| `4 DUP ADD END` | `[8]` |
| `1 9 ADD END` | `[0]` |
| `2 9 SUB END` | `[3]` |
| `3 1 4 SWAP SUB END` | `[3, 3]` |
| `5 2 DUP ADD END` | `[5, 4]` |
| `4 7 2 POP SUB END` | `[7]` |
| `2 8 SWAP POP END` | `[8]` |
| `3 5 1 SWAP DUP ADD END` | `[3, 1, 0]` |
| `6 DUP DUP ADD SUB END` | `[4]` |

## Recommended First Task

Use full final stack prediction.

Why:

- Stronger than top-of-stack prediction
- Simpler and cleaner than trace prediction
- Encourages the model to track both stack depth and multiple values
- Keeps automatic evaluation easy with exact-match metrics

Suggested task format:

```text
3 1 4 SWAP SUB END OUT 3 3 STOP
```

`OUT` and `STOP` are task-format tokens, not machine instructions. Serialize the final stack bottom to top.

## Research Caution

Good performance on this task would show that the model can predict final stacks. It would not, by itself, prove that the model stores a literal stack data structure internally.

Useful later controls:

- Compare against the same language with top-of-stack-only supervision
- Test on held-out longer programs or different depth mixtures
