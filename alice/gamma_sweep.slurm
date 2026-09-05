#!/bin/bash
#SBATCH --job-name=gsweep
#SBATCH --partition=gpu-mig-40g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-11
#SBATCH --output=%x_%A_%a.out

# gamma-sensitivity sweep (testbed plan step 2): NS-DPO at t_cp=81 for
# gamma in GAMMAS x rho_diff in RHOS. Task i -> (GAMMAS[i / |RHOS|], RHOS[i % |RHOS|]).
#   sbatch gamma_sweep.slurm <SFT_DIR>
set -e
SFT_DIR=${1:?sft run dir name}
GAMMAS=(0.3 0.5 0.7 0.9 0.95 0.99)
RHOS=(0.7 0.9)
i=$SLURM_ARRAY_TASK_ID
G=${GAMMAS[$(( i / ${#RHOS[@]} ))]}
R=${RHOS[$(( i % ${#RHOS[@]} ))]}
echo "sweep task $i -> gamma=$G rho=$R"
bash $HOME/nsdpo-scripts/ufb_dpo.slurm ns_dpo 81 $R $SFT_DIR 2021 1e-5 $G
