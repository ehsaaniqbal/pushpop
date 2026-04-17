# PP0 Phase 2 Milestone 13

This milestone tries to close the loop from a causal source head to a downstream receiver site.

## Goal

Find where the winner’s downstream top-of-stack representation degrades most after `block_0.attn_head_7` ablation, then run one small mediation check at that receiver candidate.

## Core Idea

Use a fixed long-program subset that reaches:

- `pc_15`
- `pc_16`
- `pc_17`

Part A: **Receiver search**

- zero-ablate `block_0.attn_head_7` at `pc_15`
- capture downstream hidden states at later layers / positions
- fit a baseline linear `top` probe on each candidate site
- measure how much test accuracy drops on the ablated activations

Part B: **Mediation check**

- choose the strongest candidate receiver site from Part A
- run matched-suffix head-7 patching on the same long-program subset
- compare:
  - patch only
  - receiver ablation only
  - patch plus receiver ablation

## Why This Helps

This is still not a full circuit proof, but it is the minimum credible bridge from:

- “this head matters”

to

- “this later site seems to be one place where the head’s top-relevant signal gets used”

## Metrics

For receiver search, report:

- baseline top-probe test accuracy
- ablated top-probe test accuracy
- readout drop

For mediation, report:

- source-top transfer rate
- target-top retention rate
- full-match rates

## Success Criteria

This milestone counts as a success if:

- one downstream site stands out by readout drop
- and patch transfer weakens materially when that candidate receiver is ablated

## Main Caveats

- readout collapse still measures representation accessibility, not mechanism by itself
- receiver ablation is a coarse intervention, so mediation here is suggestive rather than final
