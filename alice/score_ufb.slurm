#!/bin/bash
#SBATCH --job-name=ufb_score
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --output=%x_%A_%a.out

# Score UltraFeedback with both reward models, sharded 10 ways:
#   tasks 0-3: train split x pairrm  (4 shards)
#   tasks 4-7: train split x armorm  (4 shards)
#   task  8:   test  split x pairrm
#   task  9:   test  split x armorm
# Resumable: finished shards are skipped (score_ufb.py checks its --out file).
set -e
source $HOME/nsdpo-scripts/env.sh
cd $HOME/nsdpo-scripts

OUT=$HOME/nsdpo-data/scored
N_TRAIN_ROWS=70000       # upper bound; script clamps to the real split size
SHARD=$(( N_TRAIN_ROWS / 4 ))

T=$SLURM_ARRAY_TASK_ID
if [ $T -le 7 ]; then
    RM=$([ $T -le 3 ] && echo pairrm || echo armorm)
    IDX=$(( T % 4 ))
    python score_ufb.py --rm $RM --split train_prefs \
        --idx-start $(( IDX * SHARD )) --idx-end $(( (IDX + 1) * SHARD )) \
        --out $OUT/train_${RM}_${IDX}.pkl
else
    RM=$([ $T -eq 8 ] && echo pairrm || echo armorm)
    python score_ufb.py --rm $RM --split test_prefs --out $OUT/test_${RM}_0.pkl
fi
