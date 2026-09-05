"""MetaOptimize step-size learning for the NS-DPO trainer.

Port of the upstream reference implementation (MetaOptimize, `tinystories/HF.py`,
Sharifnassab et al., ICML 2025) with three adaptations for this codebase:
  * operates on an explicit list of trainable tensors (LoRA adapters), never on
    frozen / quantized base weights;
  * consumes gradients already accumulated in `p.grad` (the trainer microbatches
    and clips before stepping) instead of calling `autograd.grad` itself;
  * returns per-step metrics instead of writing to TensorBoard, and exposes
    `state_dict` / `load_state_dict` so the trainer's checkpointing works.
Update equations are unchanged. Scalar step size only (see design spec).
"""
import math
from typing import Dict, List

import torch


class MetaOptimizeHF:
    def __init__(self, params: List[torch.Tensor], cfg: Dict):
        self.params = list(params)
        self.cfg = dict(cfg)
        self.discount = float(cfg["discount"])
        self.epsilon = 1e-10

        if cfg["stepsize_groups"] != "scalar":
            raise NotImplementedError("only scalar step sizes are supported")

        base = cfg["alg_base"]
        if base == "RMSProp":
            self.base_update = self._rmsprop_base_update
        else:
            raise NotImplementedError(f"alg_base={base!r}")

        meta = cfg["alg_meta"]
        if meta == "fixed":
            self.meta_update = self._no_meta_update
        elif meta == "Lion":
            self.meta_update = self._lion_meta_update
        else:
            raise NotImplementedError(f"alg_meta={meta!r}")

        # beta = log(alpha): the meta-learned quantity (scalar)
        self.beta = torch.tensor(math.log(float(cfg["alpha0"])), dtype=torch.float32)
        # base-optimizer state, one entry per parameter tensor
        self.trace_base = [torch.zeros_like(p) for p in self.params]
        self.lambda_base_t = 1.0
        # condensed trace h_t (sensitivity of the parameters to beta)
        self.h = [torch.zeros_like(p) for p in self.params]
        # meta-optimizer state (scalar)
        self.momentum_meta = torch.zeros(())

    @classmethod
    def from_model(cls, model: torch.nn.Module, cfg: Dict) -> "MetaOptimizeHF":
        """Build over the model's trainable parameters only (LoRA adapters; the
        frozen / quantized base weights never enter the optimizer)."""
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("model has no trainable parameters")
        return cls(params, cfg)

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def step(self) -> Dict[str, float]:
        grads = [p.grad for p in self.params]
        missing = [i for i, g in enumerate(grads) if g is None]
        if missing:
            raise ValueError(f"{len(missing)} parameter(s) have no gradient "
                             f"(first index {missing[0]}); call backward() before step()")
        alpha = float(torch.exp(self.beta))
        h_dot_g = sum((h * g).sum() for h, g in zip(self.h, grads))
        self.base_update(grads, alpha)
        self.meta_update(h_dot_g)
        return {"meta/alpha": alpha, "meta/beta": float(self.beta),
                "meta/h_dot_g": float(h_dot_g)}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self) -> Dict:
        return {
            "cfg": dict(self.cfg),
            "beta": self.beta.clone(),
            "lambda_base_t": self.lambda_base_t,
            "trace_base": [t.clone() for t in self.trace_base],
            "h": [h.clone() for h in self.h],
            "momentum_meta": self.momentum_meta.clone(),
        }

    def load_state_dict(self, state: Dict):
        if len(state["h"]) != len(self.params):
            raise ValueError("state_dict was saved for a different parameter list")
        self.beta = state["beta"].clone()
        self.lambda_base_t = state["lambda_base_t"]
        self.trace_base = [t.clone() for t in state["trace_base"]]
        self.h = [h.clone() for h in state["h"]]
        self.momentum_meta = state["momentum_meta"].clone()

    # ------------------------------------------------------------ base updates
    def _rmsprop_base_update(self, grads, alpha):
        lam = self.cfg["normalizer_param_base"]
        wd = self.cfg["weight_decay_base"]
        self.lambda_base_t *= lam
        mu = (1 - lam) / (1 - self.lambda_base_t)
        for i, (w, g) in enumerate(zip(self.params, grads)):
            self.trace_base[i] = lam * self.trace_base[i] + g ** 2
            delta_w = alpha * (g / (mu * self.trace_base[i] + self.epsilon) ** 0.5 + wd * w)
            w.sub_(delta_w)
            self.h[i] = self.discount * (1 - wd * alpha) * self.h[i] - delta_w

    # ------------------------------------------------------------ meta updates
    def _no_meta_update(self, h_dot_g):
        return None

    def _lion_meta_update(self, h_dot_g):
        eta = self.cfg["meta_stepsize"]
        wd = self.cfg["weight_decay_meta"]
        c = self.cfg["Lion_beta2_meta"]
        rho = self.cfg["momentum_param_meta"]
        self.beta = (1 - eta * wd) * self.beta - eta * torch.sign(
            c * self.momentum_meta + (1 - c) * h_dot_g)
        self.momentum_meta = rho * self.momentum_meta + (1 - rho) * h_dot_g
