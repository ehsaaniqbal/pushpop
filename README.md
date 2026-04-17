# pushpop

`pushpop` is a small mech interp project built around `pp0`, a synthetic stack-machine task for studying what internal state a decoder-only transformer learns when it executes short programs.

This is not a benchmark project and it is not a polished research artifact. It is a small controlled setting for learning how to separate representation claims from causal claims without hiding behind scale.

> fair warning: I am still learning. I do not fully know what I am doing. If something in this repo sounds too neat, the controls matter more than the claim.

## The main question

Can a small transformer trained on a synthetic stack-machine task be shown to represent and causally use stack-like state during execution?

## Task

`pp0` is a tiny stack-machine language over digits and a small set of stack operations. A program is a short token sequence ending in `END`.

For each example, the model reads the program and predicts the final stack after execution.

One concrete example:

```text
program: 3 5 1 SWAP DUP ADD END

stack trace:
[]              -- start
[3]             -- 3
[3, 5]          -- 5
[3, 5, 1]       -- 1
[3, 1, 5]       -- SWAP
[3, 1, 5, 5]    -- DUP
[3, 1, 0]       -- ADD
[3, 1, 0]       -- END
```

So the final stack is:

```text
[3, 1, 0]
```

Internally, the supervised target is the final stack written in bottom-to-top order after `OUT`, followed by `STOP`:

```text
3 5 1 SWAP DUP ADD END OUT 3 1 0 STOP
```

So this is not generic next-token prediction on text. It is a small program-execution task where the model has to compute the final machine state.

Main evaluation metrics:

- exact full-stack accuracy: the predicted final stack must match the gold final stack exactly
- top-of-stack accuracy: only the top element of the final stack must match

## Best result so far

- winner checkpoint: `artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt`
- model: decoder-only transformer with `4` layers, `d_model=512`, `d_mlp=1024`, `8` heads, context length `32`
- final evaluation split: `artifacts/pp0_holdout/data_seed5_test10k`

| metric                    |    value |
| ------------------------- | -------: |
| exact full-stack accuracy | `0.9669` |
| top-of-stack accuracy     | `0.9773` |

## Main findings

- Linear probes suggest that depth and top-of-stack are represented online during execution, not only at the final `END` position
- `slot_1` remains partly decodable under present-only and surface-feature controls; evidence for deeper stack slots weakens substantially once `EMPTY` inflation is removed
- Mean ablation and matched-suffix patching indicate that earlier residual-stream states make causal contributions to final top-of-stack prediction
- The strongest concrete head-level result is `block_0.attn_head_7` at `pc_15`, which is both patch-sensitive and ablation-sensitive
- Token-controlled follow-up tests suggest that this head is better described as a local previous-token-conditioned mechanism than as a clean symbolic stack register
- Probe readability, causal importance, and steerability separate in this model: readable features are not automatically isolated by simple causal feature-direction edits

What this does not show yet:

- The model learned a literal symbolic stack data structure
- A cleanly isolated end-to-end stack circuit
- The full stack is robustly represented slot-by-slot
- A successful linear probe corresponds to the model’s internal readout
- Broad insights into language-model reasoning (this is intentionally a toy task)

## Run

Install deps:

```bash
uv sync
```

Build a dataset:

```bash
uv run python scripts/build_pp0_dataset.py \
  --output-dir artifacts/pp0_example \
  --train-size 100000 \
  --val-size 1000 \
  --test-size 1000
```

Train a model:

```bash
uv run python scripts/train_pp0.py \
  --data-dir artifacts/pp0_example \
  --output-dir artifacts/pp0_run
```

Evaluate a checkpoint:

```bash
uv run python scripts/eval_pp0.py \
  --data-dir artifacts/pp0_example \
  --checkpoint artifacts/pp0_run/best.pt
```

Evaluate the winner:

```bash
uv run python scripts/eval_pp0.py \
  --data-dir artifacts/pp0_holdout/data_seed5_test10k \
  --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt
```

Run the simplest Phase 2 probe:

```bash
uv run python scripts/probe_pp0.py \
  --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl \
  --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt \
  --output-json artifacts/pp0_phase2/end_probe_winner_holdout.json
```

## Docs

- [AI charter](docs/charter.md)
- [pp0 spec](docs/pp0.spec.md)
- [pp0 phase 1](docs/pp0_phase1.md)
- [pp0 phase 2](docs/pp0_phase2.md)
