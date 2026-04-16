# PP0 Phase 2 Milestone 1

This is the smallest sensible start for PP0 mechanistic work.

## Goal

Before asking which heads matter or whether an intervention breaks the model, first check whether the model's hidden states make the final machine state linearly readable.

## Fixed Corpus

Use the Phase 1 final holdout:

- `artifacts/pp0_holdout/data_seed5_test10k/test.jsonl`

For probing, split that corpus deterministically into:

- probe-train: `8000`
- probe-val: `1000`
- probe-test: `1000`

This is a probe split, not a model-selection split.

## Position And Activations

Start with the `END` token position only.

Capture hidden states from:

- embedding output
- each transformer block output
- final layer norm output

Reason: this is the smallest check for whether the model has a clean END-state representation at all.

## Probe Targets

At the `END` position, fit simple linear probes for:

- final stack depth
- top-of-stack
- `slot_1`, `slot_2`, `slot_3`

Here `slot_k` means the value `k` below the top of the stack. If that slot does not exist, use `EMPTY`.

Top-of-stack plus `slot_1..3` covers the full stack for PP0's max depth `4`.

## Controls

For every target and layer, report:

- learned linear probe accuracy
- majority-class baseline
- shuffled-label control

## Success Criteria

This milestone counts as a success if:

- depth and top decode well above both controls at later layers
- at least some below-top slots decode above control
- the pattern is stable on held-out probe-test examples

## Non-Claims

Even strong probe results do **not** show that a component causally contributes to the answer.

Probe accuracy means:

- the information is correlated with the hidden state
- the information is linearly accessible

Probe accuracy does **not** mean:

- the model literally stores a stack data structure
- the model uses that exact linear readout to compute the answer

## Immediate Next Step After This

If this milestone works, the next step is the same probe pipeline across earlier program positions so we can ask when depth, top-of-stack, and lower stack slots become available during execution.
