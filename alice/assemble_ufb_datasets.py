"""Stage 2 of UFB change-point dataset generation: merge scored shards and build
the (t_cp, rho_diff) grid of train/test pickles that `datasets=[ufb-2rm]` loads.

Paper spec (NS-DPO Sec 5.1 + appendix): 10k train / 500 test, T=101 (100 train
time steps), preferences PairRM before t_cp and ArmoRM from t_cp on; test = ArmoRM.
t_cp in {51, 66, 81}; rho_diff (fraction of train pairs where the two RMs
disagree) in {0.5, 0.7, 0.9, 0.95, 1.0}.

    python assemble_ufb_datasets.py --scored-dir $HOME/nsdpo-data/scored \
        --out-dir $HOME/nsdpo-data/ufb_2rm
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

REPO = os.path.expanduser("~/ns-dpo/LLM")
sys.path.insert(0, os.path.join(REPO, "src", "datasets"))
from ufb import create_ufb_2rm_dataset, get_varied_alignment  # noqa: E402

TCPS = [51, 66, 81]
RHOS = [0.5, 0.7, 0.9, 0.95, 1.0]
N_TRAIN, N_TEST, TIMESTEPS = 10_000, 500, 100


def merge(scored_dir: str, split: str) -> pd.DataFrame:
    """Outer-join the per-RM shard pickles for one split on the row index."""
    frames = {}
    for rm in ["pairrm", "armorm"]:
        paths = sorted(glob.glob(os.path.join(scored_dir, f"{split}_{rm}_*.pkl")))
        assert paths, f"no scored shards for {split}/{rm} in {scored_dir}"
        frames[rm] = pd.concat([pickle.load(open(p, "rb")) for p in paths]).sort_index()
    a, b = frames["pairrm"], frames["armorm"]
    assert len(a) == len(b), f"shard row mismatch: pairrm={len(a)} armorm={len(b)}"
    merged = a.join(b[["prefs_armorm", "logits_armorm"]])
    return merged.dropna()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--sample", action="store_true",
                    help="Bernoulli-sample preferences from RM logits instead of "
                         "using deterministic argmax preferences")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_pool = merge(args.scored_dir, "train")
    test_pool = merge(args.scored_dir, "test")
    n_disagree = (train_pool.prefs_pairrm != train_pool.prefs_armorm).sum()
    print(f"train pool {len(train_pool)} rows, {n_disagree} RM-disagreeing "
          f"({n_disagree/len(train_pool):.1%}); test pool {len(test_pool)} rows")

    for rho in RHOS:
        need = int(round(N_TRAIN * rho))
        if need > n_disagree:
            print(f"SKIP rho={rho}: needs {need} disagreeing rows, pool has {n_disagree}")
            continue
        np.random.seed(args.seed)
        df_train = get_varied_alignment(
            train_pool, "prefs_pairrm", "prefs_armorm",
            num_total=N_TRAIN, num_aligned=N_TRAIN - need,
        ).reset_index(drop=True)
        df_test = test_pool.sample(n=N_TEST, random_state=args.seed).reset_index(drop=True)

        for tcp in TCPS:
            np.random.seed(args.seed)
            dict_train, dict_test, raw_train, raw_test = create_ufb_2rm_dataset(
                df_train.copy(), df_test.copy(),
                timesteps=TIMESTEPS, changepoint=tcp,
                rm1="pairrm", rm2="armorm", sample=args.sample,
            )
            tag = f"tcp{tcp}_rho{rho}"
            for name, obj in [(f"{tag}_train.pkl", dict_train),
                              (f"{tag}_test.pkl", dict_test),
                              (f"{tag}_train_raw.pkl", raw_train),
                              (f"{tag}_test_raw.pkl", raw_test)]:
                with open(os.path.join(args.out_dir, name), "wb") as f:
                    pickle.dump(obj, f)
            print(f"wrote {tag}: train {len(dict_train)} test {len(dict_test)}")
    print("ASSEMBLE_DONE")


if __name__ == "__main__":
    main()
