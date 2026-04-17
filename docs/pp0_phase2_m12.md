# PP0 Phase 2 Milestone 12

This milestone is the tighter latent-state control.

## Goal

Test whether `block_0.attn_head_7` still transfers useful state when previous-token identity is fixed, and then ask which part of the previous stack state is doing the work.

## Core Idea

Reuse the winner-only matched-suffix setup at:

- query position: `pc_15`
- causal site: `block_0.attn_head_7`
- pair target: different final `top`

Partition pairs into progressively stricter buckets:

1. `same_prev_token_diff_prev_top`
   - previous token matches
   - previous top differs

2. `same_prev_token_same_prev_top_diff_prev_depth`
   - previous token matches
   - previous top matches
   - previous depth differs

3. `same_prev_token_same_prev_top_same_prev_depth_diff_prev_slot1`
   - previous token matches
   - previous top matches
   - previous depth matches
   - one-below-top differs

## Why This Helps

This turns the earlier “same previous token” control into a more interpretable hierarchy:

- if transfer survives bucket 1, then token identity alone is not enough
- if transfer survives bucket 2, then previous top alone is not enough
- if transfer survives bucket 3, then even previous top plus depth is not enough

## Metrics

For each bucket, report:

- candidate pair count
- selected pair count
- source-top transfer rate
- target-top retention rate
- summary stats for previous token / top / depth / `slot_1`

## Success Criteria

This milestone counts as a success if:

- at least one controlled bucket still shows non-trivial transfer
- or the hierarchy cleanly shows where the effect disappears

## Main Caveats

- deeper controls reduce pair count quickly
- zero or near-zero transfer in the strictest bucket can reflect either a real absence of signal or too little surviving signal to patch cleanly
