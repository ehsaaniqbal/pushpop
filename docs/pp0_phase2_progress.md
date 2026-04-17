# PP0 Phase 2 Progress Log

This file is an append-only lab note for Phase 2 work.

## 2026-04-16: Milestone 1 Baseline

Work:

- Added END-state hidden-state capture and linear probe utilities.
- Added `scripts/probe_pp0.py`.
- Added milestone spec in `docs/pp0_phase2_m1.md`.
- Added tests covering the probe path.

Verification:

- `uv run python -m unittest discover -s tests`

Outputs:

- `artifacts/pp0_phase2/end_probe_winner_holdout.json`
- `artifacts/pp0_phase2/end_probe_control_holdout.json`

Key notes:

- The winner model showed very strong END-state depth and top-of-stack decode.
- Deeper slot probes need caution because `EMPTY` dominates the labels.

## 2026-04-16: Milestone 2 Position Probe Map

Work:

- Added milestone spec in `docs/pp0_phase2_m2.md`.
- Added program-position probe utilities.
- Added `scripts/probe_pp0_positions.py`.
- Added or extended tests for position-aligned targets and script execution.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/probe_pp0_positions.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --output-json artifacts/pp0_phase2/position_probe_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/probe_pp0_positions.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --output-json artifacts/pp0_phase2/position_probe_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/position_probe_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_control_holdout.json`

Key notes:

- Position `pc_k` is aligned to stack state **after** executing token `k`.
- Results should be read together with per-position example counts and baselines.
- Early positions contain degenerate targets by construction:
  - `depth=1` at `pc_0`
  - deep slots are often `EMPTY`
  - the JSON logs this with `is_constant_label`
- The more informative region is mid/late execution, not the trivial early perfect scores.
- Winner vs control, selected best-layer test accuracies:
  - `pc_7` top: winner `0.7678`, control `0.6517`
  - `pc_9` top: winner `0.6938`, control `0.6045`
  - `pc_12` top: winner `0.5927`, control `0.4685`
  - `pc_15` top: winner `0.5050`, control `0.3654`
  - `pc_18` top: winner `0.6094`, control `0.3125`
- Mid/late `slot_1` also stays stronger in the winner:
  - `pc_12` slot_1: winner `0.6902`, control `0.4187`
  - `pc_15` slot_1: winner `0.6545`, control `0.3621`
- Depth is close to perfectly readable across most positions in both models, often already strongest at `block_0`.
- Example counts shrink at later positions because only longer programs reach them:
  - `pc_9`: `7773` examples
  - `pc_12`: `5414`
  - `pc_15`: `3053`
  - `pc_18`: `741`

## 2026-04-16: Milestone 3 Present-Only Slots And Surface Controls

Work:

- Added milestone spec in `docs/pp0_phase2_m3.md`.
- Added surface-feature lookup controls:
  - `current_token`
  - `current_token_and_remaining_length`
- Added present-only slot probe support to remove `EMPTY` inflation for `slot_1/2/3`.
- Added `scripts/probe_pp0_position_controls.py`.
- Added tests for the new helpers and the new runner.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/probe_pp0_position_controls.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --output-json artifacts/pp0_phase2/position_probe_controls_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/probe_pp0_position_controls.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --output-json artifacts/pp0_phase2/position_probe_controls_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/position_probe_controls_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_controls_control_holdout.json`

Key notes:

- The surface controls explain the trivial early positions, but they do **not** explain the mid/late winner signal.
- Winner `top` stays well above surface controls:
  - `pc_12`: probe `0.5927`, token `0.3595`, token+remaining `0.3556`
  - `pc_15`: probe `0.5050`, token `0.3123`, token+remaining `0.3223`
- Depth also stays far above surface controls and remains nearly perfect:
  - `pc_12`: probe `0.9962`, token+remaining `0.4034`
  - `pc_15`: probe `0.9801`, token+remaining `0.3920`
- `slot_1_present_only` is the cleanest non-top stack-content result:
  - winner `pc_12`: probe `0.6452`, token `0.1613`, token+remaining `0.1567`
  - winner `pc_15`: probe `0.6286`, token `0.1184`, token+remaining `0.1020`
- The winner stays much stronger than the smaller control on these clean `slot_1_present_only` checks:
  - `pc_12`: winner `0.6452`, control `0.3249`
  - `pc_15`: winner `0.6286`, control `0.2653`
- The earlier raw `slot_2` story becomes weaker after removing `EMPTY`:
  - winner raw `slot_2` at `pc_12`: `0.6730`
  - winner `slot_2_present_only` at `pc_12`: `0.4107`
- `slot_2_present_only` is still above surface controls, but it is much less decisive than `slot_1_present_only`.
- `slot_3_present_only` is weak and noisy by late positions:
  - winner `pc_15`: probe `0.1500`, token `0.1500`, token+remaining `0.1500`
  - this is not good evidence for a robust fourth-from-top representation
- Best current interpretation:
  - depth: strong and likely tracked online
  - top: strong and online
  - one-below-top (`slot_1`): strong enough to take seriously
  - deeper slots: mixed evidence, with `slot_3` currently weak

## 2026-04-16: Milestone 4 First Causal Panel

Work:

- Added milestone spec in `docs/pp0_phase2_m4.md`.
- Added residual intervention support to the model forward path.
- Added causal helpers in `pushpop/pp0_causal.py`.
- Added `scripts/causal_ablate_pp0.py`.
- Added tests for residual intervention behavior and causal script execution.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_ablate_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 7 12 15 --layers block_0 block_1 block_2 block_3 final_ln --output-json artifacts/pp0_phase2/causal_mean_ablation_winner_holdout.json --batch-size 256 --device auto`
- `uv run python scripts/causal_ablate_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --positions 7 12 15 --layers block_0 block_1 block_2 block_3 final_ln --output-json artifacts/pp0_phase2/causal_mean_ablation_control_holdout.json --batch-size 256 --device auto`

Outputs:

- `artifacts/pp0_phase2/causal_mean_ablation_winner_holdout.json`
- `artifacts/pp0_phase2/causal_mean_ablation_control_holdout.json`

Key notes:

- Intervention used: replace one residual vector at `(layer, pc_k)` with the mean vector from that same site over the evaluated subset, then rerun full rollout evaluation.
- Each position is evaluated on the subset of examples that actually reach that position:
  - `pc_7`: `9233` examples
  - `pc_12`: `5414`
  - `pc_15`: `3053`
- The strongest causal effects in both models are in earlier blocks, especially `block_0`.
- Winner, selected exact-match drops:
  - `pc_12` `block_0`: `-0.3591`
  - `pc_12` `block_1`: `-0.2355`
  - `pc_12` `block_2`: `-0.0730`
  - `pc_15` `block_0`: `-0.3721`
  - `pc_15` `block_1`: `-0.2565`
  - `pc_15` `block_2`: `-0.0714`
- Smaller control, same sites:
  - `pc_12` `block_0`: `-0.0406`
  - `pc_12` `block_1`: `-0.0249`
  - `pc_12` `block_2`: `-0.0078`
  - `pc_15` `block_0`: `-0.1828`
  - `pc_15` `block_1`: `-0.1343`
  - `pc_15` `block_2`: `-0.0400`
- Winner top-accuracy drops are also much larger than the control at the same sites:
  - `pc_12` `block_0`: winner `-0.1956`, control `-0.0209`
  - `pc_15` `block_0`: winner `-0.2244`, control `-0.0950`
- `block_3` and `final_ln` at past program positions show essentially zero effect on future answer generation in both models.
- This is an architectural lesson, not a failed experiment:
  - in a decoder-only transformer, later positions can read earlier positions through **higher** layers
  - once we are already at the final block output (`block_3`) for an earlier token, there is no later block left for future positions to use
  - so strong late-layer probes can still coexist with weak or zero causal effect from ablating those same late-layer past-position states
- Best current causal interpretation:
  - earlier residual streams carry information that later positions actually use
  - late probe-readability by itself does not guarantee causal leverage on later outputs
  - the winner appears more causally dependent on these mid-execution residual pathways than the smaller control

## 2026-04-16: Milestone 5 Matched-Suffix Activation Patching

Work:

- Added milestone spec in `docs/pp0_phase2_m5.md`.
- Added per-example rollout helpers and matched-suffix pair selection.
- Added `scripts/causal_patch_pp0.py`.
- Extended causal tests for rollout helpers and pair matching.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_patch_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 12 15 --layers block_0 block_1 block_2 block_3 final_ln --pair-target top --max-pairs 64 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_patch_winner_holdout.json`
- `uv run python scripts/causal_patch_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --positions 12 15 --layers block_0 block_1 block_2 block_3 final_ln --pair-target top --max-pairs 64 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_patch_control_holdout.json`

Outputs:

- `artifacts/pp0_phase2/causal_patch_winner_holdout.json`
- `artifacts/pp0_phase2/causal_patch_control_holdout.json`

Key notes:

- Pair matching rule is stricter than plain suffix matching:
  - same current token at `pc_k`
  - same suffix after `pc_k`
  - different top-of-stack at `pc_k`
  - both baseline rollouts already match their own gold outputs
- This gives many valid counterfactual pairs:
  - winner `pc_12`: `658584` candidates, `64` sampled
  - winner `pc_15`: `539264` candidates, `64` sampled
- Main metric is whether patching source state into the target makes the target output move toward the source.
- Winner, source-top transfer:
  - `pc_12` `block_0`: source-top `0.1250`, target-top `0.5469`
  - `pc_12` `block_1`: source-top `0.1094`, target-top `0.7188`
  - `pc_15` `block_0`: source-top `0.2500`, target-top `0.5000`
  - `pc_15` `block_1`: source-top `0.1719`, target-top `0.5625`
- Winner late-layer controls stay near zero transfer:
  - `pc_12` `block_3`: source-top `0.0000`, target-top `1.0000`
  - `pc_15` `final_ln`: source-top `0.0000`, target-top `1.0000`
- The smaller control shows much weaker transfer at the same early sites:
  - `pc_12` `block_0`: source-top `0.0312`, target-top `0.7969`
  - `pc_15` `block_0`: source-top `0.0469`, target-top `0.8438`
- Full-output transfer is much weaker than top transfer in both models:
  - example: winner `pc_15` `block_0` source-full `0.0000`, source-top `0.2500`
- Best current interpretation:
  - earlier blocks carry state that can partially steer the shared continuation toward a different final top
  - the winner supports much stronger causal top transfer than the smaller control
  - this is still partial transfer, not full stack-state transplantation

## 2026-04-16: Milestone 6 Probe-Direction Top Steering

Work:

- Added milestone spec in `docs/pp0_phase2_m6.md`.
- Added additive residual interventions.
- Added fitted ridge-probe objects that expose class-difference directions.
- Added `scripts/causal_steer_pp0.py`.
- Extended causal tests for additive intervention and steering script execution.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_steer_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 12 15 --layers block_0 block_1 block_2 block_3 --alphas 0.5 1.0 2.0 4.0 --max-pairs 32 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_steer_winner_holdout.json`
- `uv run python scripts/causal_steer_pp0.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 12 15 --layers block_0 block_1 --alphas 8 16 32 64 128 256 --max-pairs 32 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_steer_winner_holdout_highalpha.json`

Outputs:

- `artifacts/pp0_phase2/causal_steer_winner_holdout.json`
- `artifacts/pp0_phase2/causal_steer_winner_holdout_highalpha.json`

Key notes:

- Setup:
  - fit a linear `top` probe on probe-train examples
  - pick matched-suffix test pairs with different top at `pc_k`
  - add the source-minus-target probe direction to the target activation
  - compare against shuffled-label probe directions
- Pair counts were healthy:
  - `pc_12`: `5750` candidate pairs, `32` sampled
  - `pc_15`: `6146` candidate pairs, `32` sampled
- Probe quality itself was non-trivial:
  - `pc_12` `block_2` top probe test accuracy: `0.5927`
  - `pc_15` `block_3` top probe test accuracy: `0.5050`
- Result: **no measurable top steering**.
  - source-top transfer stayed `0.0000` across the tested layers and alpha sweeps
  - target-top retention stayed `1.0000`
  - shuffled-label control was equally ineffective
- This remained true even after a much larger alpha sweep focused on the causal layers:
  - alphas `8, 16, 32, 64, 128, 256`
  - still zero source-top transfer at `pc_12` / `pc_15` in `block_0` and `block_1`
- Important interpretation:
  - whole-vector patching works
  - simple linear `top`-direction steering does **not**
  - so the easy hypothesis “one probe-defined top direction is enough to drive the final top” is currently unsupported
- Best current interpretation:
  - `top` is linearly readable
  - some earlier residual state is causally useful
  - but the causally useful state is not isolated by this simple class-difference probe direction

## 2026-04-17: Milestone 7 Component-Family Causal Decomposition

Work:

- Added milestone spec in `docs/pp0_phase2_m7.md`.
- Exposed per-block attention and MLP outputs as intervention sites in the model.
- Added reusable matched-suffix site-patching helpers in `pushpop/pp0_causal.py`.
- Added component-output capture in `pushpop/pp0_probe.py`.
- Added `scripts/causal_patch_pp0_components.py`.
- Extended causal tests for component outputs and the new runner.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_patch_pp0_components.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 12 15 --blocks block_0 block_1 --components attn mlp --pair-target top --max-pairs 64 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_component_patch_winner_holdout.json`
- `uv run python scripts/causal_patch_pp0_components.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l384_1m_ft_lr1e4_bs128_2ep/best.pt --positions 12 15 --blocks block_0 block_1 --components attn mlp --pair-target top --max-pairs 64 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_component_patch_control_holdout.json`

Outputs:

- `artifacts/pp0_phase2/causal_component_patch_winner_holdout.json`
- `artifacts/pp0_phase2/causal_component_patch_control_holdout.json`

Key notes:

- This reuses the exact matched-suffix pair definition from milestone 5.
- The only change is the intervention target:
  - `block_i.attn`
  - `block_i.mlp`
- Winner, `block_0` family split:
  - `pc_12` attention source-top transfer: `0.1250`
  - `pc_12` MLP source-top transfer: `0.0625`
  - `pc_15` attention source-top transfer: `0.2500`
  - `pc_15` MLP source-top transfer: `0.0938`
- Winner, `block_1` family split is weak:
  - `pc_12` attention: `0.0000`, MLP: `0.0000`
  - `pc_15` attention: `0.0156`, MLP: `0.0156`
- Smaller control, same `block_0` comparison:
  - `pc_12` attention: `0.0312`, MLP: `0.0156`
  - `pc_15` attention: `0.0469`, MLP: `0.0156`
- Smaller control, `block_1` is again weak or near-zero:
  - `pc_12` attention: `0.0000`, MLP: `0.0000`
  - `pc_15` attention: `0.0156`, MLP: `0.0000`
- Best current interpretation:
  - the earlier reusable top-carrying state is concentrated more in the `block_0` attention update than in the `block_0` MLP update
  - the winner shows a much stronger `block_0.attn` effect than the smaller control
  - `block_1` does not show a strong family-level transfer signal under this intervention
- Important caveat:
  - “attention beats MLP” here means the attention **update** is the more causally transferable site under this patching setup
  - it does **not** by itself prove that attention is the sole storage location of the relevant state
  - a reasonable next hypothesis is that `block_0` attention is copying or routing state that later computation relies on

## 2026-04-17: Milestone 8 Winner-Only Head-Level Attention Patching

Work:

- Added milestone spec in `docs/pp0_phase2_m8.md`.
- Extended the attention module to expose per-head post-`out_proj` residual contributions.
- Added attention-head capture in `pushpop/pp0_probe.py`.
- Added `scripts/causal_patch_pp0_attention_heads.py`.
- Extended causal tests for head decomposition and the new runner.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/causal_patch_pp0_attention_heads.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 15 --blocks block_0 --pair-target top --max-pairs 64 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_head_patch_winner_pc15.json`

Outputs:

- `artifacts/pp0_phase2/causal_head_patch_winner_pc15.json`

Key notes:

- This panel is winner-only, following the project direction from this point onward.
- Setup:
  - same matched-suffix top-transfer pairs as milestone 5
  - same `pc_15` / `block_0` site that looked strongest in milestone 7
  - patch one attention head contribution at a time
- Main result:
  - `head_7` is the clear standout head
  - `head_7` source-top transfer: `0.2344`
  - `head_7` target-top retention: `0.5469`
- This is strikingly close to whole-`block_0.attn` patching at the same site:
  - whole `block_0.attn` source-top transfer: `0.2500`
  - whole `block_0.attn` target-top retention: `0.5000`
- The next strongest head is much smaller:
  - `head_1` source-top transfer: `0.0469`
- The remaining heads are effectively inert on this metric:
  - `head_0`, `head_2`, `head_3`, `head_4`, `head_5`, `head_6`: `0.0000`
- Best current interpretation:
  - most of the earlier `block_0.attn` causal top-transfer effect is concentrated in a single head, `block_0.attn_head_7`
  - this is much stronger evidence than the earlier family-level result because the whole attention effect is now nearly reproduced by one head
- Important caveat:
  - this still does not prove that `head_7` alone implements the full stack algorithm
  - it shows that `head_7` carries most of the causally transferable state for the matched-suffix top-transfer setup at this site
- Best next step:
  - inspect `block_0.attn_head_7` attention patterns and then test whether ablating or patching that head specifically changes top-of-stack more than other outputs

## 2026-04-17: Milestone 9 Dominant Head Inspection And Ablation

Work:

- Added milestone spec in `docs/pp0_phase2_m9.md`.
- Extended the attention module to return attention patterns.
- Added attention-pattern capture in `pushpop/pp0_probe.py`.
- Added zero-ablation helper in `pushpop/pp0_causal.py`.
- Added `scripts/inspect_pp0_attention_head.py`.
- Added `scripts/causal_ablate_pp0_attention_heads.py`.
- Extended causal tests for attention-pattern return, zero ablation, and the new runners.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/inspect_pp0_attention_head.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --position 15 --block block_0 --head 7 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/inspect_head7_winner_pc15.json`
- `uv run python scripts/causal_ablate_pp0_attention_heads.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --positions 15 --blocks block_0 --heads 0 1 2 3 4 5 6 7 --ablation-mode zero --batch-size 256 --device auto --output-json artifacts/pp0_phase2/causal_head_ablation_winner_pc15.json`

Outputs:

- `artifacts/pp0_phase2/inspect_head7_winner_pc15.json`
- `artifacts/pp0_phase2/causal_head_ablation_winner_pc15.json`

Key notes:

- This milestone is winner-only.
- Attention inspection for `block_0.attn_head_7` at `pc_15` was run on exact-correct examples only:
  - subset size: `2786`
  - attention rows sum to `1.0` as expected
- Main descriptive result:
  - `head_7` is overwhelmingly focused on the immediately previous source position
  - average attention to source position `14`: `0.8832`
  - average attention to the current position `15`: `0.0318`
  - all earlier source positions are much smaller
- The strongest `(source_position, token)` aggregates are all at source position `14`:
  - `pos14 POP`: total mass `466.25`
  - `pos14 SWAP`: `466.15`
  - `pos14 ADD`: `329.94`
  - `pos14 SUB`: `304.51`
  - `pos14 DUP`: `268.51`
- Conservative interpretation of the attention pattern:
  - `head_7` looks much more like a strong previous-token reader than a broad prefix search head at this site
  - this by itself does not yet say what feature it is extracting from that previous token
- Head zero-ablation panel on `pc_15` / `block_0`:
  - `head_7` is also the strongest necessity result
  - `head_7` exact-match delta: `-0.2928`
  - `head_7` top-accuracy delta: `-0.1805`
  - `head_7` token-accuracy delta: `-0.0879`
  - `head_7` stop-accuracy delta: `-0.0003`
- This is much larger than the other heads:
  - `head_1` top-accuracy delta: `-0.0973`
  - `head_5` top-accuracy delta: `-0.0583`
  - the rest are much smaller
- Best current interpretation:
  - `block_0.attn_head_7` is both the strongest causal transfer head and the strongest causal ablation head at `pc_15`
  - it appears to read mainly from the immediately previous token position
  - removing it hurts top-of-stack much more than stop prediction and substantially more than overall token accuracy
- Important caveat:
  - the head is still not cleanly “top-only,” because exact-match and token accuracy also drop
  - the safest claim is that it is especially important for the part of the computation that determines the final top, not that it is exclusively dedicated to that one feature
- Best next step:
  - inspect matched examples for what token/state lives at source position `14`
  - then test whether `head_7` is copying the previous token identity itself or using it as a pointer to a stack-relevant latent state

## 2026-04-17: Milestone 10 Previous-Position Identity vs Latent-State Test

Work:

- Added milestone spec in `docs/pp0_phase2_m10.md`.
- Added `scripts/analyze_pp0_head7_prev_position.py`.
- Extended causal tests for the new analysis script.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/analyze_pp0_head7_prev_position.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --position 15 --block block_0 --head 7 --pair-target top --max-pairs 256 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/head7_prev_position_control_winner_pc15_256.json`

Outputs:

- `artifacts/pp0_phase2/head7_prev_position_control_winner_pc15_256.json`

Key notes:

- This panel reuses the winner-only matched-suffix setup at `pc_15` for `block_0.attn_head_7`.
- It partitions candidate pairs by what is true at source position `14`:
  - `different_prev_token`
  - `same_prev_token`
  - `same_prev_token_same_prev_top`
- Candidate-pair composition:
  - all pairs: `539264`
  - different previous token: `447756`
  - same previous token: `91508`
  - same previous token and same previous top: `1564`
- Head-7 final-top transfer stays strong when the previous token differs:
  - `different_prev_token` source-top transfer: `0.2109`
  - `different_prev_token` target-top retention: `0.5156`
- But the effect collapses when the previous token is the same:
  - `same_prev_token` source-top transfer: `0.0000`
  - `same_prev_token` target-top retention: `1.0000`
- In the stricter bucket where both previous token and previous top match, only a weak residual effect remains:
  - `same_prev_token_same_prev_top` source-top transfer: `0.0547`
  - `same_prev_token_same_prev_top` target-top retention: `1.0000`
- Previous-position state statistics clarify why this bucket is interesting:
  - in `same_prev_token_same_prev_top`, previous full stack is still always different
  - previous depth is the same only `53.7%` of the time
  - so these pairs isolate deeper latent state beyond token identity and previous top
- Best current interpretation:
  - the dominant head-7 patching effect depends strongly on differences in previous-token identity at position `14`
  - this substantially weakens the simple “head-7 reads a fully token-agnostic latent stack state from position 14” story
  - there is still weak residual evidence in the `same_prev_token_same_prev_top` bucket, but it is much smaller and not yet strong enough to lean on
- Practical takeaway:
  - the head looks most like a previous-token-centered mechanism whose causal transfer is largely gated by what token sits at position `14`
  - any deeper latent-state role is, at best, secondary in this current test

## 2026-04-17: Milestone 11 Head-7 Previous-Token Category Analysis

Work:

- Added milestone spec in `docs/pp0_phase2_m11.md`.
- Added a generic pair-intervention runner in `pushpop/pp0_causal.py` so the later mediation step can reuse the same matched-pair metric logic.
- Added `scripts/analyze_pp0_head7_prev_token_types.py`.
- Extended causal tests for the new script.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/analyze_pp0_head7_prev_token_types.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --position 15 --block block_0 --head 7 --pair-target top --max-pairs-per-bucket 256 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/head7_prev_token_types_winner_pc15.json`

Outputs:

- `artifacts/pp0_phase2/head7_prev_token_types_winner_pc15.json`

Key notes:

- This is winner-only and stays at the same causal site: `block_0.attn_head_7` at `pc_15`.
- Ablation result by previous-token category at position `14`:
  - `DIGIT`: top delta `-0.3932`, exact delta `-0.6767`, count `730`
  - `ADD`: top delta `-0.3496`, exact delta `-0.5339`, count `369`
  - `SUB`: top delta `-0.3105`, exact delta `-0.4843`, count `351`
  - `DUP`: top delta `-0.0334`, exact delta `-0.0669`, count `299`
  - `POP`: top delta `-0.0190`, exact delta `-0.0274`, count `474`
  - `SWAP`: top delta `-0.0160`, exact delta `-0.0373`, count `563`
- So the strongest necessity result is not “all stack ops equally”; it is much stronger for previous `DIGIT`, `ADD`, and `SUB` than for `DUP`, `POP`, or `SWAP`.
- Patching result by source previous-token category:
  - `DIGIT`: source-top transfer `0.7539`, target-top retention `0.2188`
  - `SUB`: source-top transfer `0.1094`, target-top retention `0.5078`
  - `POP`: source-top transfer `0.0586`, target-top retention `0.6523`
  - `SWAP`: source-top transfer `0.0508`, target-top retention `0.5664`
  - `DUP`: source-top transfer `0.0469`, target-top retention `0.6289`
  - `ADD`: source-top transfer `0.0391`, target-top retention `0.6758`
- Best current interpretation:
  - `head_7` is not just “stack-op sensitive” in a uniform way
  - it is especially implicated when the previous token is a literal digit, `ADD`, or `SUB`
  - `SWAP` / `POP` / `DUP` look much weaker in this head-specific analysis
- Practical takeaway:
  - the dominant head-7 story is more “previous-token-conditioned local computation” than “generic carrier for all stack-manipulation ops”

## 2026-04-17: Milestone 12 Tighter Token-Controlled State Buckets

Work:

- Added milestone spec in `docs/pp0_phase2_m12.md`.
- Added `scripts/analyze_pp0_head7_token_controlled_state.py`.
- Extended causal tests for the new script.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/analyze_pp0_head7_token_controlled_state.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --position 15 --block block_0 --head 7 --pair-target top --max-pairs 256 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/head7_token_controlled_state_winner_pc15.json`

Outputs:

- `artifacts/pp0_phase2/head7_token_controlled_state_winner_pc15.json`

Key notes:

- This panel tightens the “same previous token” control by asking which previous-state difference, if any, keeps head-7 transfer alive.
- Results:
  - `same_prev_token_diff_prev_top`: source-top transfer `0.0000`, target-top retention `1.0000`
  - `same_prev_token_same_prev_top_diff_prev_depth`: source-top transfer `0.0430`, target-top retention `1.0000`
  - `same_prev_token_same_prev_top_same_prev_depth_diff_prev_slot1`: source-top transfer `0.0508`, target-top retention `1.0000`
- Candidate-pair counts stayed healthy even under strict controls:
  - `same_prev_token_diff_prev_top`: `89944`
  - `same_prev_token_same_prev_top_diff_prev_depth`: `724`
  - `same_prev_token_same_prev_top_same_prev_depth_diff_prev_slot1`: `840`
- Best current interpretation:
  - once previous-token identity is fixed, the strong head-7 transfer basically disappears
  - differences in previous top, previous depth, or previous `slot_1` do not recover the earlier large effect
  - the weak residual `0.043-0.051` transfer is too small to support a strong latent-state story
- Practical takeaway:
  - this is the cleanest falsification of the “generic latent stack-state reader” story
  - `head_7` looks primarily token-conditioned, with at most a weak secondary dependence on deeper state

## 2026-04-17: Milestone 13 Receiver Search And Mediation

Work:

- Added milestone spec in `docs/pp0_phase2_m13.md`.
- Extended `pushpop/pp0_probe.py` so program-position hidden-state capture can run under interventions.
- Added `scripts/analyze_pp0_head7_receiver.py`.
- Extended causal tests for the new receiver script.

Verification:

- `uv run python -m unittest discover -s tests`
- `uv run python scripts/analyze_pp0_head7_receiver.py --data-path artifacts/pp0_holdout/data_seed5_test10k/test.jsonl --checkpoint artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt --query-position 15 --positions 15 16 17 --block block_0 --head 7 --pair-target top --max-pairs 256 --batch-size 256 --device auto --output-json artifacts/pp0_phase2/head7_receiver_winner_pc15.json`

Outputs:

- `artifacts/pp0_phase2/head7_receiver_winner_pc15.json`

Key notes:

- Receiver search uses the common long-program subset that reaches `pc_15`, `pc_16`, and `pc_17`:
  - common subset size: `1543`
  - exact-correct subset for mediation: `1372`
- Largest downstream top-readout drop under `head_7` ablation is a late site:
  - `block_3 @ pc_17`: baseline `0.5161`, ablated `0.4355`, drop `0.0806`
- For mediation, the script intentionally picks the earliest near-best receiver candidate instead of that late echo:
  - candidate receiver: `block_1 @ pc_15`
  - baseline `0.3968`, ablated `0.3226`, drop `0.0742`
- Mediation panel on `block_1 @ pc_15`:
  - patch only: source-top transfer `0.1797`, target-top retention `0.6406`
  - receiver mean ablation only: source-top transfer `0.0898`, target-top retention `0.8398`
  - patch plus receiver mean ablation: source-top transfer `0.1289`, target-top retention `0.7031`
- Best current interpretation:
  - there is a plausible immediate downstream receiver at `block_1 @ pc_15`
  - ablating that site weakens the patch effect, but does not make it disappear
  - so the evidence points to partial mediation, not a clean single-site receiver story
- Practical takeaway:
  - we can now tell a narrow circuit story:
    - `block_0.attn_head_7` influences later top-relevant state immediately at `block_1 @ pc_15`
    - but the effect is not cleanly exhausted by one downstream site
  - the more honest conclusion is “partially traceable local pathway,” not “solved end-to-end circuit”
