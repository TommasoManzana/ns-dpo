#!/bin/bash
#SBATCH --job-name=sseep
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out

# Seed replicates for the gamma-sweep cells that carry the claims (plateau top and
# gamma->1 cliff): NS-DPO at t_cp=81, SEEDS x GAMMAS x RHOS.
# Task i -> seed = SEEDS[i / (|GAMMAS|*|RHOS|)], then gamma, then rho.
#   sbatch [--export=ALL,DATA=...] seed_sweep.slurm <SFT_DIR>
set -e
SFT_DIR=${1:?sft run dir name}
SEEDS=(2022 2023)
GAMMAS=(0.9 0.95 0.99)
RHOS=(0.7 0.9)
i=$SLURM_ARRAY_TASK_ID
nr=${#RHOS[@]}; ng=${#GAMMAS[@]}
S=${SEEDS[$(( i / (ng*nr) ))]}
G=${GAMMAS[$(( (i / nr) % ng ))]}
R=${RHOS[$(( i % nr ))]}
echo "seed-sweep task $i -> seed=$S gamma=$G rho=$R data=${DATA:-default}"
bash $HOME/nsdpo-scripts/ufb_dpo.slurm ns_dpo 81 $R $SFT_DIR $S 1e-5 $G
