# Source this to get the NS-DPO environment on ALICE (login or compute node).
#   source $HOME/nsdpo-scripts/env.sh
module load ALICE/default Python/3.11.3-GCCcore-12.3.0
source $HOME/nsdpo-venv/bin/activate

# Compute nodes have no outbound network: run fully from local caches there.
# (Harmless on login nodes only if you export ONLINE=1 first to re-enable downloads.)
if [ -z "$ONLINE" ]; then
    export HF_HUB_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    export WANDB_MODE=offline
fi
export HF_HOME=$HOME/.cache/huggingface
export PYTHONUNBUFFERED=1
