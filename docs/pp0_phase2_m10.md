# PP0 Phase 2 Milestone 10

This milestone tests a sharper hypothesis about the dominant winner head.

## Goal

Test whether `block_0.attn_head_7` is mainly copying previous-token identity, or whether it is using the previous position as a pointer to stack-relevant latent state.

## Core Idea

Stay with the same winner-only matched-suffix setup at:

- query position: `pc_15`
- causal site: `block_0.attn_head_7`
- pair target: different final `top`

Now partition matched pairs by what is true at the **source position `14`**:

1. `same_prev_token`
   - source and target have the same token at position `14`

2. `same_prev_token_same_prev_top`
   - source and target have the same token at position `14`
   - and the same top-of-stack **after executing token `14`**

3. `different_prev_token`
   - source and target have different tokens at position `14`

## Why This Helps

If head-7 patching still transfers top in `same_prev_token` pairs, then previous-token identity alone cannot explain the effect.

If head-7 patching still transfers top in `same_prev_token_same_prev_top` pairs, then even previous-token identity plus previous top-of-stack is not enough, which points more strongly to deeper latent stack state at position `14`.

## Metrics

For each bucket, report:

- candidate pair count
- selected pair count
- head-7 patched source-top transfer rate
- head-7 patched target-top retention rate
- whether source and target differ at position `14` on:
  - top
  - depth
  - full stack

## Success Criteria

This milestone counts as a success if:

- `same_prev_token` still shows non-trivial head-7 transfer
- ideally `same_prev_token_same_prev_top` also remains above zero
- the result rules out the simplest “copy previous token identity” story

## Main Caveats

- This still does not prove exactly what latent feature is being read from position `14`
- Zero transfer in the strongest bucket would weaken the latent-state story, but could still reflect limited pair counts
- These are matched-pair interventions, not a complete account of the full model computation

## Immediate Next Step After This

If head-7 still transfers in `same_prev_token_same_prev_top` pairs, the next step is to inspect whether the remaining difference is best explained by:

- depth
- one-below-top (`slot_1`)
- or another deeper stack feature stored at position `14`
