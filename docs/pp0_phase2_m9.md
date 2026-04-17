# PP0 Phase 2 Milestone 9

This milestone follows up on the dominant winner head found in milestone 8.

## Goal

Understand `block_0.attn_head_7` better in two ways:

- where it attends at the causal query position `pc_15`
- whether removing just that head hurts top-of-stack more than broader output metrics

## Core Idea

Use the same winner-only setting and stay focused on the strongest site:

- model: winner only
- block: `block_0`
- position: `pc_15`
- head: `7`

### Part A: Attention Inspection

Capture the attention pattern for `block_0.attn_head_7` at `pc_15` on exact-correct examples.

Report simple summaries:

- average attention by source position
- total / mean attention mass by source token identity
- a few top `(source_position, token)` aggregates

This is descriptive evidence, not proof of mechanism.

### Part B: Head-Specific Ablation

Zero-ablate the residual contribution of one head at `(block_0.attn_head_h, pc_15)` and rerun rollout evaluation.

Use a small panel over all heads, but keep the interpretation centered on `head_7`.

## Metrics

For the ablation panel, report:

- exact-match delta
- top-accuracy delta
- stop-accuracy delta
- token-accuracy delta
- loss delta

This lets us ask whether removing `head_7` hurts top-of-stack more specifically than other outputs.

## Success Criteria

This milestone counts as a success if:

- the attention summary gives a concrete picture of what `head_7` is reading from
- `head_7` also stands out under ablation, not just patching
- the top-accuracy drop is meaningfully stronger than the stop/token drops

## Main Caveats

- Attention inspection is only descriptive
- Zero ablation is a stronger and less on-manifold intervention than patching
- A head can matter for top-of-stack without being exclusively about top-of-stack

## Immediate Next Step After This

If `head_7` both attends in an interpretable way and stands out under ablation, the next step is:

- inspect the exact source positions it uses on matched examples
- then test whether this head is better described as copying, selecting, or updating a stack-like feature
