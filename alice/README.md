# ALICE testbed scripts (Leiden HPC)

Everything needed to run the NS-DPO LLM experiments on ALICE. On the cluster,
`~/nsdpo-scripts` is a symlink to this directory; all scripts assume that path.

Setup (once): `env.sh` (module + venv + offline flags), `predownload.sh` and
`predownload_traincache.sh` (stage HF assets on a login node; compute nodes are
offline — note the repo's own `cache_dir` convention, hence two caches),
`smoke_sft.slurm` (tiny-mistral smoke test).

Datasets: `score_ufb.slurm` (10-shard array: UltraFeedback scored by PairRM and
ArmoRM) → `assemble_ufb_datasets.py` (t_cp × ρ_diff grid of pickles).

Training: `ufb_sft.slurm` (SFT reference), `ufb_dpo.slurm` (one DPO / NS-DPO
run), `gamma_sweep.slurm` (γ sweep array). Recommended partition for training:
`gpu-mig-40g` (A100 MIG, the paper's hardware class), e.g.

    sbatch -p gpu-mig-40g -t 08:00:00 ufb_dpo.slurm ns_dpo 81 0.9 <SFT_DIR>

Deviations from upstream, all deliberate: `lr=1e-5` (the repo default 5e-7 does
not learn with LoRA on a 4-bit base; the paper does not document its value),
`q_proj` typo fixed in the Llama-2 LoRA config, W&B offline on compute nodes
(`wandb sync` from a login node), stray dataset dump in `ufb.py` silenced.
