#!/bin/bash
# Seed replicates. For each seed, train every config in CONFIGS into out/<config>_s<seed>, then
# evaluate its learner -- and pretrained GPT-2 -- with the same seed. Each eval is its own job,
# chained on its training job with --dependency=afterok, and writes out/seeds/<label>.json.
#
#   bash scripts/launch_seeds.sh                                   # seed 1, Table 1 configs, GLUE
#   SEEDS="1 2 3" CONFIGS="epigen frozengen ppl nobuffer" EVALS="[glue]" bash scripts/launch_seeds.sh
#
# CONFIGS may also name random_init and mixture_{0p,25p,50p,75p,100p}; a mixture run loads that
# seed's epigen generator, waiting for the epigen job when epigen is in CONFIGS and using the saved
# out/epigen_s<seed>/generator otherwise. TRAIN_SLURM / EVAL_SLURM select the sbatch scripts (default
# scripts/train.slurm and scripts/eval.slurm); CKPT sets checkpoint_interval (0 disables);
# TRAIN_EXTRA / EVAL_EXTRA append hydra overrides (e.g. TRAIN_EXTRA="no_wandb=true").
# A training job is refused when out/<config>_s<seed>/learner already exists (a finished run would be
# overwritten, and runs are not bitwise reproducible); set OVERWRITE=1 to retrain anyway.
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS=${SEEDS:-"1"}
CONFIGS=${CONFIGS:-"epigen frozengen ppl nobuffer"}
EVALS=${EVALS:-"[glue]"}
CKPT=${CKPT:-20000}
TRAIN_SLURM=${TRAIN_SLURM:-scripts/train.slurm}
EVAL_SLURM=${EVAL_SLURM:-scripts/eval.slurm}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
EVAL_EXTRA=${EVAL_EXTRA:-}
OVERWRITE=${OVERWRITE:-0}
mkdir -p logs out/seeds

if [ "$OVERWRITE" != 1 ]; then
  for seed in $SEEDS; do for cfg in $CONFIGS; do
    if [ -d "out/${cfg}_s${seed}/learner" ]; then
      echo "refusing: out/${cfg}_s${seed}/learner exists and would be overwritten (OVERWRITE=1 to retrain)" >&2
      exit 1
    fi
  done; done
fi

config_dir() {
  case $1 in
    epigen|frozengen|ppl|nobuffer) echo glue_table ;;
    random_init) echo random_init ;;
    mixture_*) echo mixture ;;
    *) echo "unknown config: $1" >&2; exit 1 ;;
  esac
}

submit_eval() { # label path seed [dependency_jid]
  local label=$1 path=$2 seed=$3 dep=${4:-}
  local dep_args=()
  [ -n "$dep" ] && dep_args+=(--dependency="afterok:$dep")
  jid=$(sbatch --parsable --job-name="eval_$label" \
    --output="logs/slurm_%x_%j.out" --error="logs/slurm_%x_%j.err" "${dep_args[@]}" \
    "$EVAL_SLURM" "runs=[{label:$label,path:$path}]" "evals=$EVALS" "seed=$seed" "output=out/seeds/$label.json" $EVAL_EXTRA)
  echo "  eval  $label -> $jid${dep:+ (after $dep)}"
}

for seed in $SEEDS; do
  epigen_jid=""
  for cfg in $CONFIGS; do
    tag="${cfg}_s${seed}"
    overrides=("seed=$seed" "tag=$tag" "out_dir=out/$tag" "+checkpoint_interval=$CKPT" $TRAIN_EXTRA)
    dep_args=()
    if [[ $cfg == mixture_* ]]; then
      overrides+=("generator_init_path=out/epigen_s${seed}/generator")
      if [ -n "$epigen_jid" ]; then
        dep_args+=(--dependency="afterok:$epigen_jid")
      elif [ ! -d "out/epigen_s${seed}/generator" ]; then
        echo "$cfg needs epigen in CONFIGS before it, or a finished out/epigen_s${seed}/generator" >&2; exit 1
      fi
    fi
    jid=$(sbatch --parsable --job-name="$tag" \
      --output="logs/slurm_%x_%j.out" --error="logs/slurm_%x_%j.err" "${dep_args[@]}" \
      "$TRAIN_SLURM" "$(config_dir "$cfg")" "$cfg" "${overrides[@]}")
    echo "train $tag -> $jid"
    [ "$cfg" = epigen ] && epigen_jid=$jid
    submit_eval "$tag" "out/$tag/learner" "$seed" "$jid"
  done
  [ -f "out/seeds/pretrained_s${seed}.json" ] || submit_eval "pretrained_s${seed}" gpt2 "$seed"
done
