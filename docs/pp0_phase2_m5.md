# PP0 Phase 2 Milestone 5

This milestone moves from site-level ablation to feature-carrying counterfactual patching.

## Goal

Test whether a chosen site carries reusable execution state that later tokens actually use.

## Core Idea

Use **matched-suffix activation patching**.

For a fixed program position `pc_k`, choose two examples that:

- reach `pc_k`
- have the **same current token** at `pc_k`
- have the **same suffix** after `pc_k`
- differ in stack state after executing token `k`

Because the future continuation is identical, the continuation should only need the state at `pc_k`, not a different future program.

Then:

- run the source example and capture one hidden state site `(layer, pc_k)`
- patch that source state into the target example at the same site
- rerun the target example rollout

If the target output moves toward the source output, that is direct evidence that the patched site carries causally useful state for the shared continuation.

## First Panel

Start with:

- model: winner only
- positions: `pc_12`, `pc_15`
- layers: `block_0`, `block_1`, `block_2`, `block_3`, `final_ln`
- pair selection target: different **top-of-stack** at `pc_k`

Only use pairs where both baseline rollouts already match their own gold outputs.

## Metrics

For each site, report:

- selected pair count
- fraction of patched targets whose final output matches the source gold output
- fraction still matching the original target gold output
- fraction whose final top matches the source top
- fraction whose final top stays at the original target top

## Success Criteria

This milestone counts as a success if:

- patching earlier causally important sites makes the target output move toward the source
- control sites like `block_3` and `final_ln` stay weak
- the effect is stronger than what we would expect from probe readability alone

## Main Caveats

- This is still site-level patching, not a single-neuron or head-level mechanism
- Same current token plus same suffix removes a major confound, but not every confound
- Failure can still mean the representation is redundant or that the chosen site is too late

## Immediate Next Step After This

If matched-suffix patching works, the next step is to narrow the intervention:

- patch only one component family at a time
- or remove / patch only the probe-defined feature direction instead of the whole residual vector
