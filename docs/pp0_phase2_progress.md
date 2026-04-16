# PP0 Phase 2 Progress Log

This file is an append-only lab note for Phase 2 work.

## 2026-04-16: Milestone 1 Baseline

Work:

- Added END-state hidden-state capture and linear probe utilities.
- Added `scripts/probe_pp0.py`.
- Added milestone spec in `docs/pp0_phase2_m1.md`.
- Added tests covering the probe path.

Verification:

- `uv run python -m unittest discover -s tests`

Outputs:

- `artifacts/pp0_phase2/end_probe_winner_holdout.json`
- `artifacts/pp0_phase2/end_probe_control_holdout.json`

Key notes:

- The winner model showed very strong END-state depth and top-of-stack decode.
- Deeper slot probes need caution because `EMPTY` dominates the labels.

## 2026-04-16: Milestone 2 Position Probe Map

Work:

- Added milestone spec in `docs/pp0_phase2_m2.md`.
- Added program-position probe utilities.
- Added `scripts/probe_pp0_positions.py`.
- Added or extended tests for position-aligned targets and script execution.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/probe_pp0_positions.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --output-json artifacts/pp0_phase2/position_probe_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/probe_pp0_positions.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --output-json artifacts/pp0_phase2/position_probe_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/position_probe_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_control_holdout.json`

Key notes:

- Position `pc_k` is aligned to stack state **after** executing token `k`.
- Results should be read together with per-position example counts and baselines.
- Early positions contain degenerate targets by construction:
  - `depth=1` at `pc_0`
  - deep slots are often `EMPTY`
  - the JSON logs this with `is_constant_label`
- The more informative region is mid/late execution, not the trivial early perfect scores.
- Winner vs control, selected best-layer test accuracies:
  - `pc_7` top: winner `0.7678`, control `0.6517`
  - `pc_9` top: winner `0.6938`, control `0.6045`
  - `pc_12` top: winner `0.5927`, control `0.4685`
  - `pc_15` top: winner `0.5050`, control `0.3654`
  - `pc_18` top: winner `0.6094`, control `0.3125`
- Mid/late `slot_1` also stays stronger in the winner:
  - `pc_12` slot_1: winner `0.6902`, control `0.4187`
  - `pc_15` slot_1: winner `0.6545`, control `0.3621`
- Depth is close to perfectly readable across most positions in both models, often already strongest at `block_0`.
- Example counts shrink at later positions because only longer programs reach them:
  - `pc_9`: `7773` examples
  - `pc_12`: `5414`
  - `pc_15`: `3053`
  - `pc_18`: `741`

## 2026-04-16: Milestone 3 Present-Only Slots And Surface Controls

Work:

- Added milestone spec in `docs/pp0_phase2_m3.md`.
- Added surface-feature lookup controls:
  - `current_token`
  - `current_token_and_remaining_length`
- Added present-only slot probe support to remove `EMPTY` inflation for `slot_1/2/3`.
- Added `scripts/probe_pp0_position_controls.py`.
- Added tests for the new helpers and the new runner.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/probe_pp0_position_controls.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --output-json artifacts/pp0_phase2/position_probe_controls_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/probe_pp0_position_controls.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --output-json artifacts/pp0_phase2/position_probe_controls_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/position_probe_controls_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_controls_control_holdout.json`

Key notes:

- The surface controls explain the trivial early positions, but they do **not** explain the mid/late winner signal.
- Winner `top` stays well above surface controls:
  - `pc_12`: probe `0.5927`, token `0.3595`, token+remaining `0.3556`
  - `pc_15`: probe `0.5050`, token `0.3123`, token+remaining `0.3223`
- Depth also stays far above surface controls and remains nearly perfect:
  - `pc_12`: probe `0.9962`, token+remaining `0.4034`
  - `pc_15`: probe `0.9801`, token+remaining `0.3920`
- `slot_1_present_only` is the cleanest non-top stack-content result:
  - winner `pc_12`: probe `0.6452`, token `0.1613`, token+remaining `0.1567`
  - winner `pc_15`: probe `0.6286`, token `0.1184`, token+remaining `0.1020`
- The winner stays much stronger than the smaller control on these clean `slot_1_present_only` checks:
  - `pc_12`: winner `0.6452`, control `0.3249`
  - `pc_15`: winner `0.6286`, control `0.2653`
- The earlier raw `slot_2` story becomes weaker after removing `EMPTY`:
  - winner raw `slot_2` at `pc_12`: `0.6730`
  - winner `slot_2_present_only` at `pc_12`: `0.4107`
- `slot_2_present_only` is still above surface controls, but it is much less decisive than `slot_1_present_only`.
- `slot_3_present_only` is weak and noisy by late positions:
  - winner `pc_15`: probe `0.1500`, token `0.1500`, token+remaining `0.1500`
  - this is not good evidence for a robust fourth-from-top representation
- Best current interpretation:
  - depth: strong and likely tracked online
  - top: strong and online
  - one-below-top (`slot_1`): strong enough to take seriously
  - deeper slots: mixed evidence, with `slot_3` currently weak

## 2026-04-16: Milestone 4 First Causal Panel

Work:

- Added milestone spec in `docs/pp0_phase2_m4.md`.
- Added residual intervention support to the model forward path.
- Added causal helpers in `pushpop/pp0_causal.py`.
- Added `scripts/causal_ablate_pp0.py`.
- Added tests for residual intervention behavior and causal script execution.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_ablate_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 7 12 15 --layers block_0 block_1 block_2 block_3 final_ln --output-json artifacts/pp0_phase2/causal_mean_ablation_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/causal_ablate_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --positions 7 12 15 --layers block_0 block_1 block_2 block_3 final_ln --output-json artifacts/pp0_phase2/causal_mean_ablation_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/causal_mean_ablation_winner_holdout.json`
- `artifacts/pp0_phase2/causal_mean_ablation_control_holdout.json`

Key notes:

- Intervention used: replace one residual vector at `(layer, pc_k)` with the mean vector from that same site over the evaluated subset, then rerun full rollout evaluation.
- Each position is evaluated on the subset of examples that actually reach that position:
  - `pc_7`: `9233` examples
  - `pc_12`: `5414`
  - `pc_15`: `3053`
- The strongest causal effects in both models are in earlier blocks, especially `block_0`.
- Winner, selected exact-match drops:
  - `pc_12` `block_0`: `-0.3591`
  - `pc_12` `block_1`: `-0.2355`
  - `pc_12` `block_2`: `-0.0730`
  - `pc_15` `block_0`: `-0.3721`
  - `pc_15` `block_1`: `-0.2565`
  - `pc_15` `block_2`: `-0.0714`
- Smaller control, same sites:
  - `pc_12` `block_0`: `-0.0406`
  - `pc_12` `block_1`: `-0.0249`
  - `pc_12` `block_2`: `-0.0078`
  - `pc_15` `block_0`: `-0.1828`
  - `pc_15` `block_1`: `-0.1343`
  - `pc_15` `block_2`: `-0.0400`
- Winner top-accuracy drops are also much larger than the control at the same sites:
  - `pc_12` `block_0`: winner `-0.1956`, control `-0.0209`
  - `pc_15` `block_0`: winner `-0.2244`, control `-0.0950`
- `block_3` and `final_ln` at past program positions show essentially zero effect on future answer generation in both models.
- This is an architectural lesson, not a failed experiment:
  - in a decoder-only transformer, later positions can read earlier positions through **higher** layers
  - once we are already at the final block output (`block_3`) for an earlier token, there is no later block left for future positions to use
  - so strong late-layer probes can still coexist with weak or zero causal effect from ablating those same late-layer past-position states
- Best current causal interpretation:
  - earlier residual streams carry information that later positions actually use
  - late probe-readability by itself does not guarantee causal leverage on later outputs
  - the winner appears more causally dependent on these mid-execution residual pathways than the smaller control
