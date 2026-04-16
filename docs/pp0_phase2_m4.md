# PP0 Phase 2 Milestone 4

This milestone begins the first targeted causal tests.

## Goal

Test whether the layer-position regions highlighted by the probes actually matter for final behavior.

## Intervention

Use a simple residual-stream intervention:

- pick one layer-position site `(layer, pc_k)`
- replace that residual vector with the mean residual vector at the same site, computed over the fixed evaluation subset
- rerun normal rollout evaluation

This is a **mean ablation**. It removes example-specific information at that site while keeping the vector on-manifold enough to avoid the harshest distribution shift from zeroing.

## Evaluation Setup

For each probed position:

- evaluate only on examples that actually reach that program position
- compute a no-intervention baseline on that same subset
- compare exact-match and top-of-stack accuracy after intervention

This avoids diluting late-position effects with examples that never reach the site.

## First Panel

Start with a small panel around the strongest probe regions:

- positions: `pc_7`, `pc_12`, `pc_15`
- layers: `block_0`, `block_1`, `block_2`, `block_3`, `final_ln`

Run the same panel for:

- the winner model
- the smaller control model

## Success Criteria

This milestone counts as a success if:

- ablating late-layer sites at `pc_12` or `pc_15` hurts the winner more than weakly motivated sites
- the winner shows a cleaner causal concentration around top / near-top probe regions than the smaller control

## What This Can And Cannot Tell Us

Positive result:

- the site causally contributes to final behavior under this intervention

Negative result:

- the site is not necessary under this intervention, or the representation is redundant

This does **not** yet identify a precise feature or head-level mechanism.

## Immediate Next Step After This

If one or two sites show clear causal importance, the next step is a more specific causal test:

- compare targeted sites against matched neighboring controls
- then move from whole-residual ablation toward more specific components
