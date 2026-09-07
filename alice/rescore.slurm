#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%A_%a.out

# Re-score saved checkpoints on the full test split (n_eval_examples=2000), no
# training: loads <run_dir>/LATEST/policy.pt as the POLICY and the SFT checkpoint
# (REF_SFT env, required) as the REFERENCE, then reads the
# "eval after 0" line (do_first_eval). Dataset (tcp/rho) is parsed from the run dir
# name; loss=dpo is used for the eval pass (reward accuracy = sign of the implicit
# reward margin, independent of the loss's gamma/beta).
#   sbatch --array=0-$((N-1)) --export=ALL,REF_SFT=<sft run dir name> rescore.slurm <list file>
# Always run task 0 alone first and check its accuracy is not 0 before the rest.
# Needs a >=40 GB GPU (two 7B models + eval): on gpu-short pass --gres=gpu:a100:1
# (L4 24 GB nodes OOM); node872's NFS mount is broken (2026-09-07): --exclude=node872.
set -e -o pipefail   # pipefail: a python failure must fail the job even through the grep filter
LIST=${1:?file listing run dir names (under ~/nsdpo-cache/s4206355)}
REF_SFT=${REF_SFT:?set REF_SFT=<sft run dir name> (the reference model)}
REF_ARCHIVE=$HOME/nsdpo-cache/s4206355/$REF_SFT/LATEST/policy.pt
[ -f "$REF_ARCHIVE" ] || { echo "missing $REF_ARCHIVE"; exit 1; }
RUN=$(sed -n "$(( SLURM_ARRAY_TASK_ID + 1 ))p" $LIST)
[ -n "$RUN" ] || { echo "no run for task $SLURM_ARRAY_TASK_ID"; exit 0; }
source $HOME/nsdpo-scripts/env.sh
REPO=${REPO:-$HOME/ns-dpo}
cd $REPO/LLM
DATA=${DATA:-$HOME/nsdpo-data/ufb_2rm_v2}
TCP=$(echo "$RUN" | grep -o "tcp[0-9]*" | head -1 | cut -c4-)
RHO=$(echo "$RUN" | grep -o "rho[0-9.]*" | head -1 | cut -c4- | sed 's/\.$//')
ARCHIVE=$HOME/nsdpo-cache/s4206355/$RUN/LATEST/policy.pt
[ -f "$ARCHIVE" ] || { echo "missing $ARCHIVE"; exit 1; }
echo "RESCORE run=$RUN tcp=$TCP rho=$RHO ref=$REF_SFT"
python3 train.py model=llama2-7b-chat-hf datasets=[ufb-2rm] \
    loss=dpo loss.beta=0.1 model.archive=$ARCHIVE ++model.reference_archive=$REF_ARCHIVE \
    exp_name=rescore_${RUN} n_epochs=null n_examples=24 eval_every=100000 \
    n_eval_examples=${N_EVAL:-2000} batch_size=24 eval_batch_size=12 \
    trainer=BasicTrainer sample_during_eval=false \
    "local_dirs=[$HOME/nsdpo-cache]" ++wandb.enabled=false ++test_dataset=false \
    ++dataset.path_train=$DATA/tcp${TCP}_rho${RHO}_train.pkl \
    ++dataset.path_test=$DATA/tcp${TCP}_rho${RHO}_test.pkl 2>&1 | grep --line-buffered -aE "eval after 0|Loaded .* eval batches|REFERENCE|Traceback|Error"
# the one-batch "training" run above leaves a 3.7 GB run dir we never want: remove it
rm -rf $HOME/nsdpo-cache/s4206355/rescore_${RUN}_*
echo "RESCORE_DONE run=$RUN"
