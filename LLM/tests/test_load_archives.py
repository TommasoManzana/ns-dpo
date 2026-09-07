"""Policy and reference may load from different checkpoints (needed to re-score a
trained policy against its SFT reference; default behaviour loads both from one)."""
import torch

from src.models import load_archives


def _save(tmp_path, name, value):
    m = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        m.weight.fill_(value)
    p = tmp_path / name
    torch.save({"step_idx": 0, "metrics": {}, "state": m.state_dict()}, p)
    return str(p)


def test_both_models_load_the_single_archive_by_default(tmp_path):
    a = _save(tmp_path, "a.pt", 1.0)
    policy, ref = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    load_archives(policy, ref, archive=a, reference_archive=None)
    assert policy.weight.item() == 1.0 and ref.weight.item() == 1.0


def test_reference_archive_loads_into_reference_only(tmp_path):
    a = _save(tmp_path, "policy.pt", 2.0)
    r = _save(tmp_path, "sft.pt", 3.0)
    policy, ref = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    load_archives(policy, ref, archive=a, reference_archive=r)
    assert policy.weight.item() == 2.0
    assert ref.weight.item() == 3.0


def test_no_archive_leaves_models_untouched(tmp_path):
    policy, ref = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        policy.weight.fill_(5.0); ref.weight.fill_(6.0)
    load_archives(policy, ref, archive=None, reference_archive=None)
    assert policy.weight.item() == 5.0 and ref.weight.item() == 6.0
