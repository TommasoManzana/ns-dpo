#!/bin/bash
#SBATCH --job-name=ufb_sft
#SBATCH --partition=gpu-a100-80g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x_%j.out

# SFT Llama-2-7b-chat (LoRA) on the UFB change-point dataset's sft_targets.
# The resulting policy.pt is the reference model for all DPO/NS-DPO runs.
# Usage:  sbatch ufb_sft.slurm [TCP] [RHO]     (default 81 / 0.9)
set -e
source $HOME/nsdpo-scripts/env.sh
cd $HOME/ns-dpo/LLM

TCP=${1:-81}
RHO=${2:-0.9}
DATA=$HOME/nsdpo-data/ufb_2rm
EXP_NAME=ufb-sft-llama2_tcp${TCP}_rho${RHO}

python3 train.py model=llama2-7b-chat-hf datasets=[ufb-2rm] \
    loss=sft exp_name=$EXP_NAME gradient_accumulation_steps=2 \
    batch_size=24 eval_batch_size=24 trainer=BasicTrainer sample_during_eval=false \
    "local_dirs=[$HOME/nsdpo-cache]" \
    ++wandb.enabled=false ++test_dataset=false \
    ++dataset.path_train=$DATA/tcp${TCP}_rho${RHO}_train.pkl \
    ++dataset.path_test=$DATA/tcp${TCP}_rho${RHO}_test.pkl

echo "SFT_DONE exp=$EXP_NAME"
echo "SFT model dir (pass to DPO scripts):"
find $HOME/nsdpo-cache -maxdepth 2 -name "*${EXP_NAME}*"
