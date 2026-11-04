#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

epigen_jid=$(sbatch --parsable scripts/train.slurm glue_table epigen)
echo "epigen -> $epigen_jid"

jids=("$epigen_jid")
for cfg in frozengen ppl nobuffer; do
  jid=$(sbatch --parsable scripts/train.slurm glue_table "$cfg")
  echo "$cfg -> $jid"
  jids+=("$jid")
done

jid=$(sbatch --parsable scripts/train.slurm random_init random_init)
echo "random_init -> $jid"
jids+=("$jid")

for cfg in mixture_25p mixture_50p mixture_100p; do
  jid=$(sbatch --parsable --dependency="afterok:$epigen_jid" scripts/train.slurm mixture "$cfg")
  echo "$cfg -> $jid (after epigen)"
  jids+=("$jid")
done

deps=$(IFS=:; echo "${jids[*]}")
jid=$(sbatch --parsable --dependency="afterok:$deps" scripts/eval.slurm)
echo "eval -> $jid (after all training)"
