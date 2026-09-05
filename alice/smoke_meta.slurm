#!/bin/bash
#SBATCH --job-name=meta_smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=%x_%j.out

# MetaOptimize smoke: tiny-mistral SFT on NSGO-2C with optimizer=MetaOptimize (few examples).
# REPO env var selects the checkout (default ~/ns-dpo; use REPO=$HOME/ns-dpo-meta for the worktree):
#   sbatch --export=ALL,REPO=$HOME/ns-dpo-meta smoke_meta.slurm
# Validates: venv on a GPU node, CUDA, dataset pipeline from local HF cache,
# LoRA + trainer save path. NOT a real training run.
set -e
source $HOME/nsdpo-scripts/env.sh
REPO=${REPO:-$HOME/ns-dpo}
cd $REPO/LLM && echo "code: $(git rev-parse HEAD) @ $REPO"

nvidia-smi -L
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python3 train.py model=tiny-mistral datasets=[nsgo-2c] \
    loss=sft optimizer=MetaOptimize lr=1e-4 exp_name=smoke-meta-tiny gradient_accumulation_steps=2 \
    batch_size=8 eval_batch_size=8 n_epochs=null n_examples=256 n_eval_examples=32 \
    trainer=BasicTrainer sample_during_eval=false \
    "local_dirs=[$HOME/nsdpo-cache]" \
    ++wandb.enabled=true ++wandb.entity=manzana-tommaso-leiden-university ++wandb.project=nsdpo ++test_dataset=false \
    ++dataset.timesteps=100 ++dataset.force_new=true ++dataset.sample_to_size=10000 \
    ++dataset.coef_shift=1.0 ++dataset.threshold_high=0.99 ++dataset.threshold_low=0.01 \
    ++dataset.country2=Germany ++dataset.min_diff=0.2

echo "META_SMOKE_OK"
