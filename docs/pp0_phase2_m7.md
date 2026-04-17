# PP0 Phase 2 Milestone 7

This milestone decomposes the earlier matched-suffix patching result by component family.

## Goal

Test whether the reusable execution state we saw in earlier blocks is carried more by:

- the block attention update
- the block MLP update

## Core Idea

Reuse the matched-suffix setup from milestone 5:

- same current token at `pc_k`
- same suffix after `pc_k`
- different top-of-stack at `pc_k`
- both baseline rollouts already match their own gold outputs

But instead of patching the full residual state at `(block_i, pc_k)`, patch only one component output:

- `block_i.attn`
- `block_i.mlp`

This asks a sharper question:

- if we only transplant the attention update, does the shared continuation move toward the source?
- if we only transplant the MLP update, does it move toward the source?

## First Panel

Keep this small and targeted:

- models: winner and smaller control
- positions: `pc_12`, `pc_15`
- blocks: `block_0`, `block_1`
- pair target: different `top`

## Metrics

For each `(position, block, component)` report:

- selected pair count
- patched source-top transfer rate
- patched target-top retention rate
- patched source-full-output transfer rate
- patched target-full-output retention rate

## Success Criteria

This milestone counts as a success if:

- at the earlier causal sites, at least one component family transfers top more than the other
- the winner shows a cleaner family-level pattern than the smaller control
- the result narrows the next intervention target

## Main Caveats

- Failure does not mean the block is irrelevant; the state may be split across both component families
- A strong attention result would not prove “attention stores the stack”; it could also mean attention routes or copies the state needed by later computation
- A strong MLP result would not prove “MLP stores the stack”; it could mean the MLP performs the state update that the continuation later relies on

## Immediate Next Step After This

If one family clearly dominates, the next step is to go one level deeper inside that family:

- attention: head-level patching
- MLP: subspace / probe-direction removal or neuron-level inspection
