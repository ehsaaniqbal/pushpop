# PP0 Phase 2 Milestone 2

This milestone extends the END-state probe into a position-by-position probe map.

## Goal

Find out when stack information becomes linearly readable during execution, not just at the end.

## Fixed Corpus

Use the same fixed holdout probe corpus as milestone 1:

- `artifacts/pp0_holdout/data_seed5_test10k/test.jsonl`

Keep the deterministic global probe split:

- probe-train: `8000`
- probe-val: `1000`
- probe-test: `1000`

## Position Definition

Probe the hidden state at each program token position `pc_k`, where:

- `k` is a `0`-based index into the PP0 program tokens, including `END`
- the hidden state is aligned with the true machine state **after** executing token `k`

This is the natural alignment for a causal transformer because the representation at position `k` can only depend on tokens up to and including `k`.

## Probe Targets

At every program position, fit the same linear probes as milestone 1:

- stack depth
- top-of-stack
- `slot_1`
- `slot_2`
- `slot_3`

Use `EMPTY` whenever the requested slot does not exist. `top` is also `EMPTY` when the stack is empty.

## Controls

For every position, target, and layer, report:

- learned linear probe accuracy
- majority-class baseline
- shuffled-label control

Also log how many examples reach that position, because late positions only exist for longer programs.

## Success Criteria

This milestone counts as a success if:

- stack information becomes more readable in later layers
- depth and top-of-stack are decodable before `END`
- the readout pattern changes across positions in ways that match execution

## Main Caveat

Absolute program positions mix together different token identities and different relative distances to `END`.

That is acceptable for the first map, but it means:

- late positions are biased toward longer programs
- deep-slot targets can still be inflated by `EMPTY`

## Immediate Next Step After This

If this milestone works, the next step is to focus on the strongest position/layer regions and run targeted causal tests there.
