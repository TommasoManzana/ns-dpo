#!/bin/bash
#SBATCH --job-name=rescore
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%A_%a.out

# Re-score saved checkpoints on the full test split (n_eval_examples=2000), no
# training: loads <run_dir>/LATEST/policy.pt as model.archive and reads the
# "eval after 0" line (do_first_eval). Dataset (tcp/rho) is parsed from the run dir
# name; loss=dpo is used for the eval pass (reward accuracy = sign of the implicit
# reward margin, independent of the loss's gamma/beta).
#   sbatch --array=0-$((N-1)) rescore.slurm <file with one run-dir name per line>
set -e
LIST=${1:?file listing run dir names (under ~/nsdpo-cache/s4206355)}
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
echo "RESCORE run=$RUN tcp=$TCP rho=$RHO"
python3 train.py model=llama2-7b-chat-hf datasets=[ufb-2rm] \
    loss=dpo loss.beta=0.1 model.archive=$ARCHIVE \
    exp_name=rescore_${RUN} n_epochs=null n_examples=24 eval_every=100000 \
    n_eval_examples=${N_EVAL:-2000} batch_size=24 eval_batch_size=12 \
    trainer=BasicTrainer sample_during_eval=false \
    "local_dirs=[$HOME/nsdpo-cache]" ++wandb.enabled=false ++test_dataset=false \
    ++dataset.path_train=$DATA/tcp${TCP}_rho${RHO}_train.pkl \
    ++dataset.path_test=$DATA/tcp${TCP}_rho${RHO}_test.pkl 2>&1 | grep -aE "eval after 0|Loaded .* eval batches|Traceback|Error"
echo "RESCORE_DONE run=$RUN"
