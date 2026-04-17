# PP0 Phase 2

Phase 2 asks what internal state the PP0 transformer learned while solving the task, not just whether it solves the task.

The main questions were:

- where stack depth is represented
- where top-of-stack is represented
- whether lower stack slots are represented
- when that information becomes available across layers and positions
- whether the model learned something stack-like or a shortcut
- which components matter causally for the final answer

## Scope

- fixed evaluation corpus: `artifacts/pp0_holdout/data_seed5_test10k/test.jsonl`
- early milestones compare the winner and the smaller control model
- from milestone 8 onward, experiments are winner-only
- all milestone code writes JSON outputs under `artifacts/pp0_phase2/`
- the standard verification step throughout was:
  - `uv run python -m unittest discover -s tests`

## Phase 2 End State

The strongest conclusions at the end of this phase are:

- the winner tracks **depth online** during execution
- the winner tracks **top-of-stack online** during execution
- `slot_1` is partly represented; deeper slots are much weaker and less reliable
- probe readability, causal importance, and steerability are **not the same thing**
- early residual pathways matter causally more than late readable states at past positions
- the strongest concrete winner mechanism found was `block_0.attn_head_7` at `pc_15`
- that head mostly reads the **immediately previous token position**
- that head is better described as a **local previous-token-conditioned mechanism** than as a clean universal stack register

Important non-claims:

- this phase does **not** show that the model learned a literal symbolic stack data structure
- this phase does **not** show that the full stack is cleanly represented slot-by-slot
- this phase does **not** isolate a single neat end-to-end “stack circuit”

## Milestone 1: END-State Linear Probes

**Date**

- `2026-04-16`

**Plan**

- start with the smallest sensible probe baseline
- use only the `END` token position
- capture hidden states from embedding output, each block output, and final layer norm
- fit linear probes for:
  - final depth
  - final top-of-stack
  - `slot_1`, `slot_2`, `slot_3`
- report majority baseline and shuffled-label control

**Implementation**

- added END-state hidden-state capture
- added linear probe utilities
- runner: `scripts/probe_pp0.py`

**Outputs**

- `artifacts/pp0_phase2/end_probe_winner_holdout.json`
- `artifacts/pp0_phase2/end_probe_control_holdout.json`

**Main result**

- the winner showed very strong END-state readout
- best held-out winner accuracies:
  - `depth`: `1.000`
  - `top`: `0.867`
  - `slot_1`: `0.703`
  - `slot_2`: `0.686`
  - `slot_3`: `0.879`
- deeper slots needed caution because `EMPTY` dominates the labels

**Takeaway**

- by `END`, final-state information is linearly readable
- this is a representation result, not a causal result

## Milestone 2: Position-By-Position Probe Map

**Date**

- `2026-04-16`

**Plan**

- extend probing from `END` to every program token position `pc_k`
- align each position to the true machine state **after** executing token `k`
- keep the same probe targets as milestone 1

**Implementation**

- added position-aligned probe utilities
- runner: `scripts/probe_pp0_positions.py`

**Outputs**

- `artifacts/pp0_phase2/position_probe_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_control_holdout.json`

**Main result**

- depth became readable very early
- top-of-stack was readable well before `END`
- selected best-layer top accuracies:
  - `pc_7`: winner `0.7678`, control `0.6517`
  - `pc_9`: winner `0.6938`, control `0.6045`
  - `pc_12`: winner `0.5927`, control `0.4685`
  - `pc_15`: winner `0.5050`, control `0.3654`
  - `pc_18`: winner `0.6094`, control `0.3125`
- selected best-layer `slot_1` accuracies:
  - `pc_12`: winner `0.6902`, control `0.4187`
  - `pc_15`: winner `0.6545`, control `0.3621`
- late positions had fewer examples:
  - `pc_9`: `7773`
  - `pc_12`: `5414`
  - `pc_15`: `3053`
  - `pc_18`: `741`

**Takeaway**

- the winner is not just reconstructing the answer at `END`
- it carries state-like information online during execution

## Milestone 3: Present-Only Slot Probes And Surface Controls

**Date**

- `2026-04-16`

**Plan**

- answer two obvious objections to the position map:
  - deeper-slot scores may be inflated by `EMPTY`
  - probe wins may come from cheap surface cues
- rerun slot probes only when the slot actually exists
- add two surface-feature baselines:
  - `current_token`
  - `current_token_and_remaining_length`

**Implementation**

- added present-only slot targets
- added surface-feature lookup controls
- runner: `scripts/probe_pp0_position_controls.py`

**Outputs**

- `artifacts/pp0_phase2/position_probe_controls_winner_holdout.json`
- `artifacts/pp0_phase2/position_probe_controls_control_holdout.json`

**Main result**

- surface controls explained trivial early positions, but not the mid/late winner signal
- winner `top` stayed well above controls:
  - `pc_12`: probe `0.5927`, token `0.3595`, token+remaining `0.3556`
  - `pc_15`: probe `0.5050`, token `0.3123`, token+remaining `0.3223`
- depth stayed nearly perfect and far above controls:
  - `pc_12`: probe `0.9962`, token+remaining `0.4034`
  - `pc_15`: probe `0.9801`, token+remaining `0.3920`
- `slot_1_present_only` was the cleanest non-top stack-content result:
  - `pc_12`: probe `0.6452`, token `0.1613`, token+remaining `0.1567`
  - `pc_15`: probe `0.6286`, token `0.1184`, token+remaining `0.1020`
- deeper-slot story weakened sharply:
  - raw `slot_2` at `pc_12`: `0.6730`
  - `slot_2_present_only` at `pc_12`: `0.4107`
  - `slot_3_present_only` at `pc_15`: `0.1500`, equal to the controls

**Takeaway**

- depth is strong
- top is strong
- `slot_1` is strong enough to take seriously
- deeper slots are mixed at best

## Milestone 4: Mean-Ablation Causal Panel

**Date**

- `2026-04-16`

**Plan**

- move from readout to causal relevance
- replace one residual vector at `(layer, pc_k)` with the mean vector from that same site
- rerun full rollout evaluation on examples that reach that position
- first panel:
  - positions: `pc_7`, `pc_12`, `pc_15`
  - layers: `block_0`, `block_1`, `block_2`, `block_3`, `final_ln`

**Implementation**

- added residual intervention support
- added causal helpers in `pushpop/pp0_causal.py`
- runner: `scripts/causal_ablate_pp0.py`

**Outputs**

- `artifacts/pp0_phase2/causal_mean_ablation_winner_holdout.json`
- `artifacts/pp0_phase2/causal_mean_ablation_control_holdout.json`

**Main result**

- the strongest causal effects were in earlier blocks, especially `block_0`
- winner exact-match drops:
  - `pc_12` `block_0`: `-0.3591`
  - `pc_12` `block_1`: `-0.2355`
  - `pc_12` `block_2`: `-0.0730`
  - `pc_15` `block_0`: `-0.3721`
  - `pc_15` `block_1`: `-0.2565`
  - `pc_15` `block_2`: `-0.0714`
- winner top drops were also large:
  - `pc_12` `block_0`: `-0.1956`
  - `pc_15` `block_0`: `-0.2244`
- `block_3` and `final_ln` at past positions had essentially zero effect

**Takeaway**

- earlier residual streams carry information that later positions actually use
- late probe readability does not imply late causal leverage on future outputs

## Milestone 5: Matched-Suffix Activation Patching

**Date**

- `2026-04-16`

**Plan**

- test whether a site carries reusable execution state
- choose source/target pairs that:
  - reach `pc_k`
  - have the same current token at `pc_k`
  - have the same suffix after `pc_k`
  - differ in top-of-stack at `pc_k`
  - are both baseline-correct
- patch source activations into the target and rerun the target

**Implementation**

- added per-example rollout helpers
- added matched-suffix pair selection
- runner: `scripts/causal_patch_pp0.py`

**Outputs**

- `artifacts/pp0_phase2/causal_patch_winner_holdout.json`
- `artifacts/pp0_phase2/causal_patch_control_holdout.json`

**Main result**

- winner source-top transfer:
  - `pc_12` `block_0`: `0.1250`
  - `pc_12` `block_1`: `0.1094`
  - `pc_15` `block_0`: `0.2500`
  - `pc_15` `block_1`: `0.1719`
- winner late controls stayed near zero:
  - `pc_12` `block_3`: `0.0000`
  - `pc_15` `final_ln`: `0.0000`
- the smaller control was much weaker at the same early sites
- full-output transfer stayed much weaker than final-top transfer

**Takeaway**

- earlier sites carry reusable state that can partially steer the shared continuation toward a different final top
- this is partial transfer, not full stack transplantation

## Milestone 6: Probe-Direction Top Steering

**Date**

- `2026-04-16`

**Plan**

- test a sharper intervention than whole-vector patching
- fit a linear `top` probe at a site
- use the source-minus-target probe direction as an additive edit
- compare against shuffled-label probe directions

**Implementation**

- added additive residual interventions
- added fitted ridge-probe objects exposing class-difference directions
- runner: `scripts/causal_steer_pp0.py`

**Outputs**

- `artifacts/pp0_phase2/causal_steer_winner_holdout.json`
- `artifacts/pp0_phase2/causal_steer_winner_holdout_highalpha.json`

**Main result**

- no measurable top steering
- source-top transfer stayed `0.0000` across the tested layers and alpha sweeps
- target-top retention stayed `1.0000`
- shuffled-label control was equally ineffective
- this remained true even with a much larger alpha sweep on `block_0` and `block_1`

**Takeaway**

- whole-vector patching works
- simple linear top-direction steering does not
- readable state and directly steerable state are not the same thing

## Milestone 7: Component-Family Causal Decomposition

**Date**

- `2026-04-17`

**Plan**

- split the earlier matched-suffix patching result by component family
- compare patching:
  - `block_i.attn`
  - `block_i.mlp`
- focus on `pc_12` and `pc_15` in `block_0` and `block_1`

**Implementation**

- exposed per-block attention and MLP outputs as intervention sites
- added component-output capture
- runner: `scripts/causal_patch_pp0_components.py`

**Outputs**

- `artifacts/pp0_phase2/causal_component_patch_winner_holdout.json`
- `artifacts/pp0_phase2/causal_component_patch_control_holdout.json`

**Main result**

- winner `block_0` attention beat winner `block_0` MLP:
  - `pc_12`: attention `0.1250`, MLP `0.0625`
  - `pc_15`: attention `0.2500`, MLP `0.0938`
- `block_1` was weak in both families
- the smaller control showed the same direction but much weaker

**Takeaway**

- the earlier reusable top-carrying state is more concentrated in the `block_0` attention update than in the `block_0` MLP update
- this does not mean attention alone “stores the stack”; it means attention was the stronger causal site under this patching setup

## Milestone 8: Winner-Only Head-Level Attention Patching

**Date**

- `2026-04-17`

**Plan**

- go one level deeper inside the strongest site from milestone 7
- patch one `block_0` attention head at a time at `pc_15`
- keep the matched-suffix final-top transfer setup

**Implementation**

- exposed per-head post-`out_proj` residual contributions
- added attention-head capture
- runner: `scripts/causal_patch_pp0_attention_heads.py`

**Outputs**

- `artifacts/pp0_phase2/causal_head_patch_winner_pc15.json`

**Main result**

- `head_7` was the clear standout
- `head_7` source-top transfer: `0.2344`
- whole `block_0.attn` source-top transfer at the same site was `0.2500`
- next strongest head was much smaller:
  - `head_1`: `0.0469`
- the other six heads were effectively inert on this metric

**Takeaway**

- most of the earlier `block_0.attn` causal top-transfer effect is concentrated in one head:
  - `block_0.attn_head_7`

## Milestone 9: Dominant Head Inspection And Ablation

**Date**

- `2026-04-17`

**Plan**

- inspect where `block_0.attn_head_7` attends at `pc_15`
- zero-ablate each `block_0` head at `pc_15`
- compare exact, top, token, and stop effects

**Implementation**

- extended attention module to return attention patterns
- added attention-pattern capture
- added head-specific zero-ablation helper
- runners:
  - `scripts/inspect_pp0_attention_head.py`
  - `scripts/causal_ablate_pp0_attention_heads.py`

**Outputs**

- `artifacts/pp0_phase2/inspect_head7_winner_pc15.json`
- `artifacts/pp0_phase2/causal_head_ablation_winner_pc15.json`

**Main result**

- `head_7` was overwhelmingly focused on the immediately previous source position
  - average attention to source position `14`: `0.8832`
  - average attention to current position `15`: `0.0318`
- `head_7` was also the strongest ablation result
  - exact delta: `-0.2928`
  - top delta: `-0.1805`
  - token delta: `-0.0879`
  - stop delta: `-0.0003`

**Takeaway**

- `head_7` is both a strong transfer head and a strong necessity head
- it looks much more like a strong previous-token reader than a broad prefix-search head

## Milestone 10: Previous-Position Identity Vs Latent-State Test

**Date**

- `2026-04-17`

**Plan**

- test whether `head_7` is mostly copying previous-token identity or reading deeper state from the previous position
- partition matched pairs by what is true at position `14`:
  - `different_prev_token`
  - `same_prev_token`
  - `same_prev_token_same_prev_top`

**Implementation**

- runner: `scripts/analyze_pp0_head7_prev_position.py`

**Outputs**

- `artifacts/pp0_phase2/head7_prev_position_control_winner_pc15_256.json`

**Main result**

- when the previous token differed, head-7 transfer stayed strong:
  - source-top transfer `0.2109`
  - target-top retention `0.5156`
- when the previous token matched, the strong effect disappeared:
  - `same_prev_token` source-top transfer `0.0000`
  - `same_prev_token` target-top retention `1.0000`
- when previous token and previous top both matched, only a weak residual effect remained:
  - source-top transfer `0.0547`
  - target-top retention `1.0000`

**Takeaway**

- the dominant head-7 patching effect depends heavily on previous-token identity
- this weakens the simple “generic latent stack-state reader” story

## Milestone 11: Head-7 Previous-Token Category Analysis

**Date**

- `2026-04-17`

**Plan**

- sharpen the previous-token story by grouping examples by the token category at position `14`
- categories:
  - `DIGIT`
  - `ADD`
  - `SUB`
  - `SWAP`
  - `POP`
  - `DUP`
- run both ablation buckets and patching buckets

**Implementation**

- added reusable pair-intervention runner support
- runner: `scripts/analyze_pp0_head7_prev_token_types.py`

**Outputs**

- `artifacts/pp0_phase2/head7_prev_token_types_winner_pc15.json`

**Main result**

- head-7 ablation was strongest when the previous token was:
  - `DIGIT`: top delta `-0.3932`
  - `ADD`: top delta `-0.3496`
  - `SUB`: top delta `-0.3105`
- the same head was much weaker for:
  - `DUP`: `-0.0334`
  - `POP`: `-0.0190`
  - `SWAP`: `-0.0160`
- patching by source previous-token category was dominated by `DIGIT`:
  - source-top transfer `0.7539`

**Takeaway**

- `head_7` is not a uniform “all stack ops” head
- it is especially implicated for local computations involving previous `DIGIT`, `ADD`, and `SUB`

## Milestone 12: Tighter Token-Controlled State Buckets

**Date**

- `2026-04-17`

**Plan**

- tighten the previous-token control further
- keep previous-token identity fixed and then vary:
  - previous top
  - previous depth
  - previous `slot_1`

**Implementation**

- runner: `scripts/analyze_pp0_head7_token_controlled_state.py`

**Outputs**

- `artifacts/pp0_phase2/head7_token_controlled_state_winner_pc15.json`

**Main result**

- once previous-token identity was fixed, the large head-7 effect basically disappeared
- source-top transfer:
  - `same_prev_token_diff_prev_top`: `0.0000`
  - `same_prev_token_same_prev_top_diff_prev_depth`: `0.0430`
  - `same_prev_token_same_prev_top_same_prev_depth_diff_prev_slot1`: `0.0508`

**Takeaway**

- this was the cleanest falsification of the “generic latent stack-state reader” story
- at most, head 7 shows a weak secondary dependence on deeper state

## Milestone 13: Receiver Search And Mediation

**Date**

- `2026-04-17`

**Plan**

- trace a downstream receiver for the head-7 signal
- first find where top-readout quality drops most after head-7 ablation
- then run one mediation check at the best early candidate

**Implementation**

- extended position-aligned hidden-state capture to work under interventions
- runner: `scripts/analyze_pp0_head7_receiver.py`

**Outputs**

- `artifacts/pp0_phase2/head7_receiver_winner_pc15.json`

**Main result**

- largest downstream top-readout drop:
  - `block_3 @ pc_17`: baseline `0.5161`, ablated `0.4355`, drop `0.0806`
- earliest near-best receiver candidate chosen for mediation:
  - `block_1 @ pc_15`: baseline `0.3968`, ablated `0.3226`, drop `0.0742`
- mediation at `block_1 @ pc_15`:
  - patch only: source-top transfer `0.1797`, target-top retention `0.6406`
  - receiver ablation only: source-top transfer `0.0898`, target-top retention `0.8398`
  - patch plus receiver ablation: source-top transfer `0.1289`, target-top retention `0.7031`

**Takeaway**

- there is a plausible immediate downstream receiver at `block_1 @ pc_15`
- ablating it weakens the patch effect, but does not remove it
- the honest story is partial mediation, not a clean single-site circuit

## Final Readout

The narrowest credible Phase 2 research claim is:

- the PP0 winner maintains stack-relevant state online, especially for depth and top-of-stack
- the strongest concrete mechanistic target found was `block_0.attn_head_7` at `pc_15`
- that head matters causally, but behaves more like a local previous-token-conditioned mechanism than a clean symbolic stack register

If this work is resumed later, the right next step is not more broad fishing. It is either:

- a small mechanistic case-study writeup centered on `head_7`, or
- turning the existing results into an interactive artifact

