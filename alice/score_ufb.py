"""Score UltraFeedback response pairs with one reward model (PairRM or ArmoRM).

Stage 1 of the UFB change-point dataset generation (NS-DPO paper Sec 5.1 / App).
Run per-shard on a GPU node; shards are merged by assemble_ufb_datasets.py.

    python score_ufb.py --rm pairrm --split train_prefs --idx-start 0 --idx-end 15000 \
        --out $HOME/nsdpo-data/scored/train_pairrm_0.pkl
"""
import argparse
import os
import pickle
import sys

import datasets as hf_datasets
import pandas as pd

REPO = os.path.expanduser("~/ns-dpo/LLM")
sys.path.insert(0, os.path.join(REPO, "src", "datasets"))  # rms.py / ufb.py import style
import rms  # noqa: E402
from ufb import polish_responses, clean_ufb  # noqa: E402


def load_ufb_split(split: str) -> pd.DataFrame:
    ds = hf_datasets.load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    df = ds.to_pandas()[["prompt", "chosen", "rejected"]]
    df = polish_responses(df)
    df = clean_ufb(df).dropna().reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rm", choices=["pairrm", "armorm"], required=True)
    ap.add_argument("--split", default="train_prefs", choices=["train_prefs", "test_prefs"])
    ap.add_argument("--idx-start", type=int, default=0)
    ap.add_argument("--idx-end", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        print(f"{args.out} already exists, nothing to do")
        return

    df = load_ufb_split(args.split)
    end = args.idx_end if args.idx_end is not None else len(df)
    shard = df.iloc[args.idx_start:end].copy()
    print(f"split={args.split} rows={len(df)} shard=[{args.idx_start}:{end}] rm={args.rm}")

    if args.rm == "pairrm":
        model, tok = rms.load_pairrm("llm-blender/PairRM-hf")
        prefs, logits = rms.apply_pairrm(model, tok, shard, "prompt", "chosen", "rejected")
    else:
        model, tok = rms.load_armorm()
        prefs, logits = rms.apply_armoRM(model, tok, shard, "prompt", "chosen", "rejected")

    shard[f"prefs_{args.rm}"] = prefs
    shard[f"logits_{args.rm}"] = logits

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(shard, f)
    print(f"SCORING_DONE {args.out} ({len(shard)} rows)")


if __name__ == "__main__":
    main()
