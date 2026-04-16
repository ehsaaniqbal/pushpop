# PP0 Phase 2 Milestone 3

This milestone strengthens the position-probe story before any causal intervention work.

## Goal

Answer two obvious objections to the position map:

- are deeper-slot scores inflated by `EMPTY`?
- are probe wins mostly recoverable from cheap surface cues like the current token and program length?

## Fixed Corpus

Use the same fixed holdout probe corpus and deterministic split as milestones 1 and 2:

- `artifacts/pp0_holdout/data_seed5_test10k/test.jsonl`
- probe-train: `8000`
- probe-val: `1000`
- probe-test: `1000`

## Analysis 1: Present-Only Slot Probes

For each program position `pc_k`, rerun the slot probes on the subset where that slot actually exists:

- `slot_1_present_only`: examples with depth at least `2`
- `slot_2_present_only`: examples with depth at least `3`
- `slot_3_present_only`: examples with depth at least `4`

This removes `EMPTY` from the label set and tests whether stack content is still linearly readable when the slot is genuinely present.

## Analysis 2: Surface-Feature Controls

For each position and target, report cheap lookup baselines that do **not** use model activations:

- `current_token`: majority label conditioned only on the token at `pc_k`
- `current_token_and_remaining_length`: majority label conditioned on the token at `pc_k` and the remaining distance to `END`

These controls are intentionally simple. They test whether easy surface structure already explains the target well.

## Success Criteria

This milestone counts as a success if:

- `slot_1_present_only` remains clearly above surface controls at informative mid/late positions
- the winner still beats the smaller control on `top` and `slot_1_present_only`
- the earlier “full stack” story becomes narrower and more honest for deeper slots

## Main Caveats

- Present-only subsets shrink fast at late positions, especially for `slot_3`
- Beating these surface controls does **not** prove a causal stack mechanism
- Failing a present-only slot probe does **not** prove the slot is absent; it only means it is not linearly accessible under this probe setup

## Immediate Next Step After This

If the winner still shows clean top / near-top signals after these controls, then targeted causal work is justified:

- choose a small number of layer-position regions
- intervene there first
- keep every causal claim paired with a falsification test
