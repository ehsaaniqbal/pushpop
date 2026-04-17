# PP0 Phase 2 Milestone 8

This milestone goes one level deeper inside the winner model's strongest causal site.

## Goal

Test whether the causal `block_0.attn` effect is concentrated in one or a few individual attention heads.

## Core Idea

Reuse the matched-suffix setup from milestone 5:

- same current token at `pc_k`
- same suffix after `pc_k`
- different top-of-stack at `pc_k`
- both baseline rollouts already match their own gold outputs

But instead of patching the whole `block_0.attn` update, patch only one head contribution at a time:

- `block_0.attn_head_0`
- `block_0.attn_head_1`
- ...

Each head contribution is defined as that head's post-`out_proj` residual contribution to the block attention update.

## First Panel

Keep this narrow:

- model: winner only
- position: `pc_15`
- block: `block_0`
- pair target: different `top`
- heads: all `8`

## Metrics

For each head, report:

- selected pair count
- patched source-top transfer rate
- patched target-top retention rate
- patched source-full-output transfer rate
- patched target-full-output retention rate

## Success Criteria

This milestone counts as a success if:

- one or a small number of heads clearly exceeds the others on source-top transfer
- the best head result is meaningfully smaller than whole-`block_0.attn` patching if the effect is distributed
- the result narrows the next mechanistic inspection target

## Main Caveats

- A weak per-head result does not mean the head is irrelevant; the effect may depend on multiple heads jointly
- A strong head result would still not prove that the head "stores" top-of-stack; it may route or compute part of the needed state
- Head contributions are measured after output projection, so this is a causal decomposition of the residual update, not a complete story of internal attention dynamics

## Immediate Next Step After This

If one head stands out, inspect that head's attention pattern and test whether patching or ablating it specifically changes top-of-stack more than depth or other outputs.
