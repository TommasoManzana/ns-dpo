#!/bin/bash
# Pre-stage the assets train.py loads through the repo's OWN cache convention
# (models.py passes cache_dir=get_local_dir(local_dirs) = <prefix>/<user>, which
# bypasses the default HF cache). Our SLURM scripts override
# local_dirs=[$HOME/nsdpo-cache], so populate $HOME/nsdpo-cache/$USER here.
# Run on a LOGIN node. Idempotent.
set -e
export ONLINE=1
source $HOME/nsdpo-scripts/env.sh

python - <<'EOF'
import getpass, os
from huggingface_hub import snapshot_download
import datasets

cache = os.path.expanduser(f"~/nsdpo-cache/{getpass.getuser()}")
os.makedirs(cache, exist_ok=True)

for repo in ["openaccess-ai-collective/tiny-mistral",
             "NousResearch/Llama-2-7b-chat-hf"]:
    print(f"--- {repo} -> {cache}")
    snapshot_download(repo, cache_dir=cache)

print(f"--- Anthropic/llm_global_opinions -> {cache}")
datasets.load_dataset("Anthropic/llm_global_opinions", cache_dir=cache)
print("TRAINCACHE_OK")
EOF
