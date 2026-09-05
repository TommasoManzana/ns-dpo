#!/bin/bash
# Pre-stage every HF asset the NS-DPO experiments need, on a LOGIN node
# (compute nodes have no outbound network). Idempotent; re-run freely.
set -e
export ONLINE=1
source $HOME/nsdpo-scripts/env.sh

python - <<'EOF'
from huggingface_hub import snapshot_download
import datasets

# Policy models (NousResearch mirror is ungated - no token needed)
for repo in [
    "openaccess-ai-collective/tiny-mistral",   # smoke-test model
    "NousResearch/Llama-2-7b-chat-hf",         # main policy model (~13 GB)
    "llm-blender/PairRM-hf",                   # reward model 1 (UFB drift, before t_cp)
    "RLHFlow/ArmoRM-Llama3-8B-v0.1",           # reward model 2 (UFB drift, after t_cp; ~16 GB)
]:
    print(f"--- {repo}")
    snapshot_download(repo)

# Datasets
print("--- Anthropic/llm_global_opinions (NSGO)")
datasets.load_dataset("Anthropic/llm_global_opinions")
print("--- HuggingFaceH4/ultrafeedback_binarized (UFB)")
datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized")
print("ALL DOWNLOADS OK")
EOF
