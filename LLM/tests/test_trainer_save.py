"""BasicTrainer.save() must work without an LR scheduler (optimizer=MetaOptimize)."""
import os

import torch

from src.metaoptimize import MetaOptimizeHF
from src.trainers import BasicTrainer


def test_save_without_scheduler_writes_policy_and_optimizer_only(tmp_path):
    trainer = BasicTrainer.__new__(BasicTrainer)   # skip the heavy __init__
    trainer.policy = torch.nn.Linear(2, 1)
    trainer.optimizer = MetaOptimizeHF.from_model(trainer.policy, dict(
        alg_base="RMSProp", normalizer_param_base=0.99, weight_decay_base=0.0,
        alg_meta="fixed", alpha0=1e-3, discount=1.0, stepsize_groups="scalar"))
    trainer.scheduler = None
    trainer.example_counter = 7
    trainer.run_dir = str(tmp_path)
    trainer.rank = 0

    trainer.save(output_dir=str(tmp_path), metrics={})

    assert os.path.exists(tmp_path / "policy.pt")
    assert os.path.exists(tmp_path / "optimizer.pt")
    assert not os.path.exists(tmp_path / "scheduler.pt")
    assert "beta" in torch.load(tmp_path / "optimizer.pt")["state"]
