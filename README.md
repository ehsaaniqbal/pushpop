# pushpop

Minimal mechanistic interpretability playground for tiny stack-machine programs.

docs:
- [Charter](/Users/ehsaan/Documents/lab/pushpop/docs/charter.md)

## Training Log

Current best checkpoints:

- Best absolute test exact: `0.8634` from `artifacts/pp0_scaleup/run_4l384_100k_scratch_ft_lr1e4`
- Best smaller-model test exact: `0.8600` from `artifacts/pp0_scaleup/run_from_best_data100k_lr1e4_plus8ep`

Notes:

- `test exact` = greedy-rollout full final-stack exact match on the held-out test split
- `test top` = greedy-rollout top-of-stack accuracy on the held-out test split
- `20k/2k/2k` is the baseline dataset in `artifacts/pp0_baseline/data`
- `100k/5k/5k` is the scale-up dataset in `artifacts/pp0_scaleup/data_seed1_100k`

| Phase    | Run                                    | Data         | Model      | Train plan                        | Test exact | Test top | Note                             |
| -------- | -------------------------------------- | ------------ | ---------- | --------------------------------- | ---------: | -------: | -------------------------------- |
| Baseline | `run_seed0`                            | `20k/2k/2k`  | `2L x 128` | `15ep, lr=3e-4`                   |   `0.1955` | `0.5270` | first working baseline           |
| Baseline | `run_seed0_wide`                       | `20k/2k/2k`  | `2L x 256` | `20ep, lr=3e-4`                   |   `0.3565` | `0.5980` | width helped a lot               |
| Baseline | `run_seed0_wide_sched`                 | `20k/2k/2k`  | `2L x 256` | `20ep, warmup_cosine`             |   `0.3450` | `0.5990` | scheduler did not help           |
| Baseline | `run_seed0_wide_sched03`               | `20k/2k/2k`  | `2L x 256` | `20ep, warmup_cosine, min_lr=0.3` |   `0.3495` | `0.6070` | still below fixed LR             |
| Baseline | `run_seed0_deep3`                      | `20k/2k/2k`  | `3L x 256` | `20ep, lr=3e-4`                   |   `0.4220` | `0.6550` | depth beat scheduler tweaks      |
| Baseline | `run_seed0_deep3_wide384_sched03`      | `20k/2k/2k`  | `3L x 384` | `20ep, warmup_cosine`             |   `0.4060` | `0.6660` | wider + scheduler, not best      |
| Baseline | `run_seed0_deep3_wide384_fixed`        | `20k/2k/2k`  | `3L x 384` | `20ep, lr=3e-4`                   |   `0.4085` | `0.6565` | more width alone was not enough  |
| Baseline | `run_seed0_deeper4_fixed`              | `20k/2k/2k`  | `4L x 256` | `20ep, lr=3e-4`                   |   `0.3620` | `0.6280` | extra depth alone was worse here |
| Baseline | `run_seed0_deep3_40ep_fixed`           | `20k/2k/2k`  | `3L x 256` | `40ep, lr=3e-4`                   |   `0.6045` | `0.7670` | longer training was huge         |
| Baseline | `run_seed0_deep3_40ep_ft_lr1e4`        | `20k/2k/2k`  | `3L x 256` | `resume +10ep, lr=1e-4`           |   `0.6375` | `0.7905` | low-LR polish helped             |
| Scale-up | `run_from_best_data100k_lr1e4_12ep`    | `100k/5k/5k` | `3L x 256` | `resume +12ep, lr=1e-4`           |   `0.8420` | `0.9012` | big data jump                    |
| Scale-up | `run_from_best_data100k_lr1e4_plus8ep` | `100k/5k/5k` | `3L x 256` | `resume +8ep, lr=1e-4`            |   `0.8600` | `0.9124` | strong small-model endpoint      |
| Scale-up | `run_4l384_100k_scratch_10ep`          | `100k/5k/5k` | `4L x 384` | `10ep from scratch, lr=3e-4`      |   `0.7026` | `0.8236` | bigger model was undertrained    |
| Scale-up | `run_4l384_20k_to100k_12ep`            | `100k/5k/5k` | `4L x 384` | `20k pretrain -> 100k +12ep`      |   `0.6770` | `0.8032` | staged branch underperformed     |
| Scale-up | `run_4l384_100k_scratch_plus10ep`      | `100k/5k/5k` | `4L x 384` | `resume +10ep, lr=3e-4`           |   `0.8346` | `0.8994` | more training mattered           |
| Scale-up | `run_4l384_100k_scratch_ft_lr1e4`      | `100k/5k/5k` | `4L x 384` | `resume +8ep, lr=1e-4`            |   `0.8634` | `0.9150` | current best                     |
