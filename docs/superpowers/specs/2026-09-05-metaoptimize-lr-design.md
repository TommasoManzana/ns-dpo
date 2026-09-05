# MetaOptimize-on-lr for NS-DPO — design (2026-09-05)

**Goal:** replace the fixed learning rate of the NS-DPO LoRA training loop with a
MetaOptimize-learned step size (scalar first), reusing the pristine upstream
implementation's math unchanged. Engineering warm-up for the thesis (meeting
2026-08-26, decision 3); the β/γ meta-updates are a separate, later track.

**Reference:** `MetaOptimize/tinystories/HF.py` (upstream, Sharifnassab et al.,
ICML 2025) — base update (SGD/SGDm/RMSProp/AdamW/Lion), condensed trace
`h ← discount·(1 − wd·α)·h − Δw`, meta-gradient `⟨h, g⟩`, meta update on
β = log α (RMSProp/Adam/Lion/fixed).

## Components

1. `LLM/src/metaoptimize.py` — `MetaOptimizeHF(params, cfg)`
   - `params`: explicit list of trainable tensors (LoRA adapters). Frozen /
     quantized base weights never enter the optimizer.
   - Same knobs and update equations as upstream: `alg_base`, `normalizer_param_base`,
     `momentum_param_base`, `weight_decay_base`, `Lion_beta2_base`; `alg_meta`,
     `meta_stepsize`, `normalizer_param_meta`, `momentum_param_meta`,
     `weight_decay_meta`, `Lion_beta2_meta`; `alpha0`, `discount`
     (upstream's γ — renamed to avoid NS-DPO's γ), `stepsize_groups`
     (`scalar` or list of group sizes).
   - `step() -> dict`: reads `p.grad` (already accumulated over microbatches and
     clipped by the trainer), applies base + meta update, returns
     `{"meta/alpha", "meta/beta", "meta/h_dot_g"}` (scalar case; per-block lists
     otherwise). Raises if any param has `grad is None`.
   - `zero_grad()`, `state_dict()`, `load_state_dict()` so the trainer's
     `save()` path keeps working. Resume is out of scope.
2. `LLM/src/trainers.py` (`BasicTrainer.train`)
   - `config.optimizer == "MetaOptimize"` → build `MetaOptimizeHF` over
     `[p for p in policy.parameters() if p.requires_grad]`; no LR scheduler.
   - Per batch: microbatch `.backward()` (unchanged) → `clip_gradient()`
     (unchanged) → `optimizer.step()` → merge returned metrics into
     `batch_metrics` → `zero_grad()`.
   - Torch optimizers keep the existing path byte-for-byte.
3. `LLM/config/config.yaml` — `meta:` block with defaults mirroring the baseline
   (base RMSProp, normalizer 0.99, wd 0) and the upstream README example for the
   meta side (Lion, meta_stepsize 1e-3, momentum 0.99, beta2 0.9, wd 0);
   `alpha0: ${lr}` (run A), `discount: 1.0`, `stepsize_groups: scalar`.
4. `alice/ufb_meta.slurm` — `ufb_dpo.slurm` with `optimizer=MetaOptimize` and an
   `ALPHA0` argument; W&B receives `meta/*` through the existing logging.

## Testing

- Unit (pytest, CPU, tiny models; run on the ALICE login node):
  1. `alg_meta=fixed` keeps α = α₀ across steps.
  2. base RMSProp step matches a hand-computed normalized-gradient step.
  3. quadratic with too-small α₀: β increases over steps (meta-gradient sign).
  4. frozen params are excluded; missing grads raise.
  5. `state_dict` round-trips to an equivalent optimizer.
- Integration: tiny-mistral smoke job with `optimizer=MetaOptimize`.
- Experiment (separate approval): run A (α₀=1e-5) vs the γ-sweep's fixed-lr
  γ=0.95 cell at t_cp=81, ρ=0.9; then run B (α₀=1e-6).

## Out of scope

Blockwise groups on Llama-2, meta-learning β/γ, resuming meta state, FSDP trainer.
