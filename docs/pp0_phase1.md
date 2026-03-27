# PP0 Phase 1

Phase 1 is done.

This was the "just train the damn thing until it actually works" phase. No mech interp yet. The goal was simple:

- build a clean PP0 training pipeline
- figure out what actually helps
- get to a model that genuinely solves the task well enough to be worth interpreting

## Final Winner

Official Phase 1 winner:

- checkpoint: `artifacts/pp0_scaleup/run_4l512_1m_scratch_ft_lr1e4_1ep/best.pt`
- model: `4L x 512`
- `d_mlp=1024`
- heads: `8`
- context length: `32`
- training data: `1,000,000` train examples from `artifacts/pp0_scaleup/data_seed4_1m`

Training path:

1. `1` epoch from scratch at `lr=3e-4`
2. `+2` epochs continuation at `lr=3e-4`
3. `+1` epoch low-LR polish at `lr=1e-4`

What it got on the fresh final holdout:

- exact full-stack accuracy: `0.9669`
- top-of-stack accuracy: `0.9773`

Short version: this thing works.

## Final Holdout

Fresh untouched final holdout:

- dataset: `artifacts/pp0_holdout/data_seed5_test10k`
- size: `10,000`
- seed: `5`

| Candidate                           | Model      | Test exact | Test top | Note                    |
| ----------------------------------- | ---------- | ---------: | -------: | ----------------------- |
| `run_4l384_300k_ft_lr1e4_bs128_4ep` | `4L x 384` |   `0.9259` | `0.9503` | old real-test champion  |
| `run_4l384_1m_ft_lr1e4_bs128_2ep`   | `4L x 384` |   `0.9546` | `0.9685` | strong smaller finalist |
| `run_4l512_1m_scratch_ft_lr1e4_1ep` | `4L x 512` |   `0.9669` | `0.9773` | Phase 1 winner          |

## What Actually Worked

Here is the blunt version.

### 1. More training mattered a lot

Early on we were underrating the models because we just had not trained them enough.

- `3L x 256` at `20` epochs: `0.4220` test exact
- same model at `40` epochs: `0.6045`
- then low-LR polish: `0.6375`

So one big lesson was: the early weak results were not just "bad models". A lot of it was straight-up undertraining.

### 2. More data mattered even more

This was the biggest clean win.

- `100k`-scale branch got us into the mid/high `0.8`s
- `300k` real-test winner hit `0.9248`
- `1M` finalists pushed us into the mid/high `0.95+` range on the final holdout

This is the most bitter-lesson part of the whole project: once the setup was sound, scaling data kept paying.

### 3. Bigger models did help, but only after they got a fair shot

The `4L x 512` model looked bad after one epoch. That was misleading.

- `1ep` scratch val exact: `0.7295`
- `+2ep` continuation: `0.8888`
- `+1ep` low-LR polish: `0.9691` on val
- final holdout: `0.9669`

So the conclusion is not "bigger models are magic". The conclusion is:

- bigger models can win
- but if you train them like crap, they look bad
- you have to actually let them cook

### 4. Low-LR polish was real

The final low-LR continuation was not placebo. More than once it gave a real bump after the higher-LR run had already done the heavy lifting.

## What Did Not Work As Well

### 1. Scheduler fiddling was weak early

Warmup/cosine did not move the needle nearly as much as:

- depth
- more epochs
- more data

That does not mean schedulers are useless. It just means they were not the main bottleneck here.

### 2. A staged bigger-model branch underperformed

One of the early "pretrain on smaller data, then move to bigger data" branches for `4L x 384` did worse than the simpler scratch branch.

Translation: not every more-complicated training story is a better one.

### 3. Width alone was not a silver bullet

Some wider early runs did not beat the best smaller/deeper branch. Width only became worth it once we also gave the model enough data and training time.

## What We Learned

This is the part I would actually keep in my head.

### The task was not wrong

At a few points it was tempting to think the problem setup was the issue.

It was not.

PP0 full final-stack prediction turned out to be a good task. The real issue was that we were initially trying to solve it with undertrained or underfed models.

### The obvious experiments were the right ones

This project worked because we did the boring sane loop:

- change one thing
- measure it
- log it
- keep the result even if it was negative

That is not "unsophisticated". That is the work.

### Validation and test are not the same thing

We kept needing to remind ourselves of this:

- validation is for steering
- test is for closing

That is why the final seed-5 holdout mattered. Without it, we would have ended Phase 1 on a validation win, which would have been sloppy.

### Researcher overfitting is real

Even if the model never trains on the test set, you can still overfit to it as a researcher by peeking too often.

The fix is simple:

- keep a dev split
- keep a final holdout
- do the final holdout once at the end

## Compact Run Log

This is the short path from "barely working" to "good enough to matter".

| Run                                    | Data           | Model      | Train plan               | Test exact | Note                            |
| -------------------------------------- | -------------- | ---------- | ------------------------ | ---------: | ------------------------------- |
| `run_seed0`                            | `20k/2k/2k`    | `2L x 128` | `15ep, lr=3e-4`          |   `0.1955` | first working baseline          |
| `run_seed0_wide`                       | `20k/2k/2k`    | `2L x 256` | `20ep, lr=3e-4`          |   `0.3565` | width helped                    |
| `run_seed0_deep3`                      | `20k/2k/2k`    | `3L x 256` | `20ep, lr=3e-4`          |   `0.4220` | depth helped more               |
| `run_seed0_deep3_40ep_fixed`           | `20k/2k/2k`    | `3L x 256` | `40ep, lr=3e-4`          |   `0.6045` | more training was huge          |
| `run_seed0_deep3_40ep_ft_lr1e4`        | `20k/2k/2k`    | `3L x 256` | `+10ep, lr=1e-4`         |   `0.6375` | low-LR polish helped            |
| `run_from_best_data100k_lr1e4_plus8ep` | `100k/5k/5k`   | `3L x 256` | continued on bigger data |   `0.8600` | first strong small-model result |
| `run_4l384_300k_ft_lr1e4_bs128_4ep`    | `300k/10k/10k` | `4L x 384` | continued on more data   |   `0.9248` | first really strong test result |
| `run_4l384_1m_ft_lr1e4_bs128_2ep`      | final holdout  | `4L x 384` | `1M` branch finalist     |   `0.9546` | strong smaller finalist         |
| `run_4l512_1m_scratch_ft_lr1e4_1ep`    | final holdout  | `4L x 512` | `1M` branch winner       |   `0.9669` | final winner                    |

## Future Things We Could Try

Not now. Later.

### 1. Longer programs

The current PP0 range tops out at length `18`. We could push length and see where the current winner breaks.

### 2. Deeper stacks

Right now max stack depth is `4`. Bumping that up should stress the hidden-state bookkeeping in a more interesting way.

### 3. Cleaner OOD tests

Examples:

- train on shorter programs, test on longer ones
- train on depth `<= 4`, test on depth `5+`
- hold out certain op mixes during training

That would tell us more about algorithmic generalization instead of just IID performance.

### 4. Smaller interpretable control models

The `4L x 512` model won, but the `4L x 384` model is still strong and easier to reason about. It may be the better first interp target even though it is not the absolute best performer.

### 5. Alternative readouts

We stuck with the causal next-token formulation because it kept the setup simple. Later we could compare that against a structured END-state readout.

### 6. Distribution tweaks

We could deliberately oversample:

- longest programs
- max-depth programs
- nasty op mixes like `ADD+DUP+SUB+SWAP`

That would answer whether targeted data shaping beats dumb scaling in this setup.

### 7. Bigger models

Obvious but true. We could keep scaling size. The main question would be whether the extra score is worth the extra interp pain.

