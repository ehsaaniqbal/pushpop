# PP0 Phase 2 Milestone 6

This milestone tests whether a probe-defined top-of-stack feature direction can *steer* model behavior.

## Goal

Move from whole-vector causal patching to a more specific intervention:

- fit a linear `top` probe at a chosen site
- use the probe to define a source-vs-target `top` direction
- add that direction to the target activation
- test whether the final top moves toward the source

## Intervention

For one site `(layer, pc_k)`:

1. fit a linear `top` probe on the probe-train split
2. for a matched source/target pair, define the direction:
   - `direction = top_class_direction[source_top] - top_class_direction[target_top]`
3. normalize the direction
4. scale it by the training-set projection standard deviation at that site
5. add `alpha * direction` to the target residual vector

This is a cleaner test than whole-vector patching because it targets the feature we think matters.

## Pair Selection

Reuse the matched-suffix setup from milestone 5:

- same current token at `pc_k`
- same suffix after `pc_k`
- different top-of-stack at `pc_k`
- both examples are baseline-correct

Use only test-split examples for the pair pool.

## First Panel

Start with the winner model only.

- positions: `pc_12`, `pc_15`
- layers: `block_0`, `block_1`, `block_2`, `block_3`
- alphas: `0.5`, `1.0`, `2.0`, `4.0`

Report both:

- true top-probe steering
- shuffled-label probe steering as a control

## Metrics

For each site, alpha, and direction type, report:

- source-top transfer rate
- target-top retention rate
- source-full-output transfer rate
- target-full-output retention rate

## Success Criteria

This milestone counts as a success if:

- true probe steering increases source-top transfer at earlier causal sites
- shuffled-label steering stays much weaker
- late-layer controls remain weak

## Main Caveats

- Success means the probe direction is causally useful under this intervention, not that it is the only representation of `top`
- Partial top transfer is already informative; full-stack flipping is a stronger bar
- Large alpha can cause off-manifold effects, so the sweep matters

## Immediate Next Step After This

If true probe directions steer behavior better than controls, the next step is:

- project out the same direction
- or isolate the component family that carries it
