# PP0 Phase 2 Milestone 11

This milestone asks whether the dominant head-7 effect is tied to specific previous-token types.

## Goal

Test whether `block_0.attn_head_7` matters mainly when the previous token at position `14` is a stack-manipulating operation, or whether its effect is similarly strong for literal digits.

## Core Idea

Stay at the same winner-only site:

- query position: `pc_15`
- causal site: `block_0.attn_head_7`

Then split the analysis by the **token category at source position `14`**:

- `ADD`
- `SUB`
- `SWAP`
- `POP`
- `DUP`
- `DIGIT`

Run two small panels:

1. **Ablation by previous-token category**
   - zero-ablate `head_7`
   - evaluate only examples whose token at position `14` falls in that category

2. **Patching by source previous-token category**
   - keep the matched-suffix setup
   - only use pairs where the previous token differs exactly between source and target
   - group pairs by the source previous-token category

## Why This Helps

If the effect is concentrated on stack-manipulating ops, that supports an op-conditioned computation story.

If digits are equally strong, then the head is more likely to be a generic previous-token mechanism.

## Metrics

For ablation buckets, report:

- example count
- exact/top/token/stop deltas

For patch buckets, report:

- candidate pair count
- selected pair count
- source-top transfer rate
- target-top retention rate

## Success Criteria

This milestone counts as a success if:

- the head-7 effect is visibly uneven across previous-token categories
- the result sharpens the story beyond “it reads the previous token”

## Main Caveats

- token categories are still coarse; a strong `DIGIT` bucket does not mean all digits behave the same
- patching buckets remain pair-distribution dependent
