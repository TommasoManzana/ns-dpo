#!/bin/bash
#SBATCH --job-name=ufb_dpo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x_%j.out

# NS-DPO or vanilla DPO on one UFB change-point dataset (Llama-2-7b-chat, LoRA).
# Usage:
#   sbatch [-p PARTITION -t TIME] ufb_dpo.slurm <ns_dpo|dpo> <TCP> <RHO> <SFT_DIR> [SEED] [LR] [GAMMA]
# Defaults: seed 2021, lr 1e-5 (repo default 5e-7 does NOT learn with LoRA), gamma 0.95.
# Partition/time are given on the sbatch command line; see README.md for the
# recommended values (gpu-mig-40g 8h is enough).
# Env overrides: WANDB=0 disables W&B (default on, offline; sync from a login node).
set -e
source $HOME/nsdpo-scripts/env.sh
cd $HOME/ns-dpo/LLM

LOSS=${1:?loss (ns_dpo|dpo)}
TCP=${2:?tcp}
RHO=${3:?rho}
SFT_DIR=${4:?sft run dir name}
SEED=${5:-2021}
LR=${6:-1e-5}
GAMMA=${7:-0.95}
WANDB=${WANDB:-1}
DATA=${DATA:-$HOME/nsdpo-data/ufb_2rm}   # override with --export=ALL,DATA=... (e.g. ufb_2rm_v2: 2k-pair test sets)

if [ "$LOSS" = "ns_dpo" ]; then
    LOSS_ARGS="loss=ns_dpo loss.gamma=$GAMMA loss.current_time=101 loss.beta=0.1"
    EXP_NAME=ufb-ns_dpo-g${GAMMA}_tcp${TCP}_rho${RHO}_s${SEED}_lr${LR}
else
    LOSS_ARGS="loss=dpo loss.beta=0.1"
    EXP_NAME=ufb-dpo_tcp${TCP}_rho${RHO}_s${SEED}_lr${LR}
fi
[ "$WANDB" = "1" ] && WB="++wandb.enabled=true ++wandb.entity=manzana-tommaso-leiden-university ++wandb.project=nsdpo" \
                   || WB="++wandb.enabled=false"

ARCHIVE=$(find $HOME/nsdpo-cache -maxdepth 4 -path "*${SFT_DIR}*/LATEST/policy.pt" | head -1)
[ -n "$ARCHIVE" ] || { echo "no policy.pt found for SFT dir '$SFT_DIR'"; exit 1; }
echo "code: $(git rev-parse HEAD) | SFT archive: $ARCHIVE | exp: $EXP_NAME"
nvidia-smi -L

python3 train.py model=llama2-7b-chat-hf datasets=[ufb-2rm] \
    seed=$SEED lr=$LR $LOSS_ARGS \
    model.archive=$ARCHIVE \
    exp_name=$EXP_NAME gradient_accumulation_steps=2 eval_every=1000 \
    batch_size=24 eval_batch_size=12 trainer=BasicTrainer sample_during_eval=false \
    "local_dirs=[$HOME/nsdpo-cache]" n_eval_examples=${N_EVAL:-2000} \
    $WB ++test_dataset=false \
    ++dataset.path_train=$DATA/tcp${TCP}_rho${RHO}_train.pkl \
    ++dataset.path_test=$DATA/tcp${TCP}_rho${RHO}_test.pkl

echo "DPO_DONE exp=$EXP_NAME"
