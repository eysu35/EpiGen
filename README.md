# EpiGen

[![arXiv](https://img.shields.io/badge/arXiv-2608.11746-b31b1b.svg)](https://arxiv.org/abs/2608.11746)

Code for [**Epiplexity Guided Data Selection and Generation for Out-of-Distribution
Generalization**](https://arxiv.org/abs/2608.11746) — epiplexity-maximizing synthetic data generation
for pretraining.

EpiGen trains a GPT-2 **generator** with REINFORCE to produce synthetic text that maximizes the
**epiplexity** of a separate GPT-2 **learner**. A
generation buffer accumulates every batch the generator has ever produced. 

At iteration `t` the generator emits a batch `X_t`, the learner takes `K` steps on it, and the reward
is the resulting drop in learner loss across the buffer:

```
r_t = sum_{x in buffer} [ log 1/P_{theta_{t-1}}(x) - log 1/P_{theta_t}(x) ]
```

That reward is the only signal the generator gets. It ascends `J = E[r_t]` by REINFORCE, with the
advantage centred on an EMA baseline `b <- beta*b + (1-beta)*r_t`:

```
grad J = E_{X_t ~ P_g} [ (r_t - b) * sum_{x in X_t} grad log P_g(x) ]
```

Concretely, one iteration of `datagen/train.py` samples `X_t` from the generator, measures the
learner's loss on `M` buffer batches plus `X_t`, takes `K` learner steps on `X_t`, re-measures to get
`r_t`, steps the generator on `(r_t - b)`, updates `b`, and appends `X_t` to the buffer.

## In this repo

```
configs/
  glue_table/       Experiment 1: epigen, frozengen, ppl, nobuffer (Table 1)
  random_init/      Experiment 2: random vs. pretrained initialization
  mixture/          Experiment 3: 0/25/50/75/100% real OpenWebText in learner training
  eval.yaml         checkpoints to score, and which evals to run on them
datagen/
  train.py          REINFORCE loop, generation, real-data mixing
  reward.py         epiplexity / no-buffer / pretrained-ppl rewards
  buffer.py         generation buffer
  eval.py           GLUE fine-tuning, perplexity, harness
scripts/
  train.slurm  eval.slurm  launch_all.sh  launch_seeds.sh
```

Every run is GPT-2 124M, `H = 6000` iterations of `S = 10` learner steps at a global batch of
32x512 tokens. Generator and learner both start from pretrained GPT-2 except in `random_init`, where
both start from scratch.

| config | reward | generator | learner data |
| --- | --- | --- | --- |
| `epigen` | epiplexity over the buffer | trained | synthetic |
| `frozengen` | epiplexity (logged only) | frozen | synthetic |
| `ppl` | negative pretrained-GPT-2 loss | trained | synthetic |
| `nobuffer` | loss drop on the current batch only | trained | synthetic |
| `random_init` | epiplexity over the buffer | trained | synthetic |
| `mixture_{0p,25p,50p,75p,100p}` | epiplexity (logged only) | frozen `out/epigen/generator` | 0/25/50/75/100% real |

Paper symbols map to config keys as `T -> H`, `K -> S`, `B -> batch_size`, `M -> sample_from_buffer`
(counted in *batches*, not sequences), `tau -> tau`, `beta -> beta`, `eta_l -> lr_learner`,
`eta_g -> lr_generator`.

## Setup

```bash
conda create -n epigen python=3.10 -y && conda activate epigen
pip install -r requirements.txt
export WANDB_API_KEY=...        # or pass no_wandb=true
```

## Data

Nothing to build ahead of time. The first run streams OpenWebText from HuggingFace, tokenizes it with
the GPT-2 tokenizer, and caches flat token arrays under `cache/` (`OWT_CACHE_DIR` for training,
`cache_dir` for eval); later runs read the cache instead of re-streaming. Held-out evaluation starts
`eval_offset` tokens into the stream, and the mixture runs draw their real training data from the
first `real_train_tokens` — startup asserts `eval_offset >= real_train_tokens` so the two never
overlap.

## Train

```bash
torchrun --nproc_per_node=2 -m datagen.train \
  --config-path ../configs/glue_table --config-name epigen
```

Swap in any of `frozengen`, `ppl`, `nobuffer`; `--config-path ../configs/random_init --config-name
random_init`; or `--config-path ../configs/mixture --config-name mixture_{0p,25p,50p,75p,100p}`. The
mixture runs load `out/epigen/generator`, so run `epigen` first. Any key can be overridden Hydra-style
(`H=100 no_wandb=true`).

`bash scripts/launch_all.sh` submits the four `glue_table` runs, `random_init` and
`mixture_{25p,50p,100p}` to SLURM, then the evaluation job once they finish (`scripts/train.slurm`
requests 2 GPUs for 30 h).

`out_dir` receives `learner/` and `generator/` in HuggingFace format, decoded text samples under
`samples/` at every eval, and periodic `learner_step*/` checkpoints if you set `checkpoint_interval`.
Hyperparameters live in each config: AdamW with weight decay 0.01 and gradient clipping at 1.0,
`lr_learner 1e-5`, `lr_generator 1e-6`, `tau 1.0`, `beta 0.99`, `sample_from_buffer 16`.

**Compute.** Roughly 24 h per run on 2 GPUs.

### Seeds

The paper reports mean and standard deviation over training seeds. `scripts/launch_seeds.sh` runs
the seeded version of the pipeline; by default it submits seed 1 of the four Table 1 configs as an
example:

```bash
bash scripts/launch_seeds.sh                       # seed 1: epigen, frozengen, ppl, nobuffer
SEEDS="1 2 3" bash scripts/launch_seeds.sh         # the full replicate set
```

Each config trains into `out/<config>_s<seed>` with `seed=<seed>`, and a dependent job then
fine-tunes that learner on GLUE with the same seed, writing `out/seeds/<config>_s<seed>.json`; a
pretrained-GPT-2 eval per seed is submitted alongside. `CONFIGS` may name any config in the table
(mixture runs use that seed's `out/epigen_s<seed>/generator`), `EVALS` picks the evals
(`"[glue,harness,openwebtext,slimpajama,fineweb]"`), `CKPT` sets `checkpoint_interval`, and
`TRAIN_EXTRA` / `EVAL_EXTRA` append Hydra overrides such as `no_wandb=true`. The launcher refuses to
retrain into an existing `out/<config>_s<seed>/learner` unless `OVERWRITE=1`.

## Evaluate

```bash
python -m datagen.eval
```

One pass scores every checkpoint in `configs/eval.yaml` on GLUE fine-tuning, the zero-shot LM Eval
Harness tasks, and held-out perplexity over OpenWebText, SlimPajama, and FineWeb. Trim the `evals`
list to run a subset. Results accumulate in `out/eval_results.json`; missing checkpoints and
already-scored runs are skipped, so the job is resumable (`force=true` re-runs everything,
`only=<label>` picks one). GLUE follows the standard protocol — 3 epochs, lr 2e-5, batch size 32,
MCC for CoLA and accuracy elsewhere. Fine-tuning is seeded by `seed` (default 42): the classifier
head and the data order are fixed per task, so re-scoring a checkpoint with the same seed is
reproducible up to GPU nondeterminism. For the mixture runs set `owt_offset=200_000_000` so the OpenWebText perplexity slice
starts past their real training pool, matching `eval_offset` in those configs.

## Citation

```bibtex
@article{su2026epiplexity,
  title   = {Epiplexity Guided Data Selection and Generation for Out-of-Distribution Generalization},
  author  = {Su, Ellen and Potapczynski, Andres and Qiu, Shikai and Hughes, Edward and Wilson, Andrew Gordon},
  journal = {arXiv preprint arXiv:2608.11746},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.11746},
}
```
