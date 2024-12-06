import os, sys
import re
import ast
import math
import json
import pickle
import datasets
import transformers
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

# small_lms = ['starchat', 'llama-2-7b-chat', 'mpt-30b-chat', 'wizardlm-7b', 'falcon-40b-instruct', 'pythia-12b', 'alpaca-7b',  'llama-2-13b-chat', 'wizardlm-13b', 'vicuna-33b', 'ultralm-13b']
small_lms = ['starchat', 'llama-2-7b-chat', 'wizardlm-7b', 'pythia-12b', 'alpaca-7b',  'llama-2-13b-chat', 'wizardlm-13b', 'ultralm-13b']
# large_lms = ['wizardlm-70b', 'llama-2-70b-chat', 'gpt-3.5-turbo', 'ultralm-65b', 'gpt-4', 'bard']
# large_lms = ['llama-2-70b-chat', 'gpt-3.5-turbo', 'gpt-4', 'bard']
large_lms = ['gpt-4']

if __name__ == "__main__":
    cache_dir=".cache/ubuntu/"
    dataset = datasets.load_dataset(
        "openbmb/UltraFeedback",
        cache_dir=cache_dir
    )

    df_orig = dataset["train"].to_pandas()

    res = list()
    for i in range(df_orig.shape[0]):
        df_target = df_orig.iloc[i]

        llms_i = [item for item in df_target["completions"] if item["model"] in large_lms]
        slms_i = [item for item in df_target["completions"] if item["model"] in small_lms]

        if len(slms_i) > 0 and len(llms_i) > 0:
            for llm_i in llms_i:
                for slm_i in slms_i:
                    res.append(
                        {
                            "instruction": df_target["instruction"],
                            "model_large": llm_i["model"],
                            "response_large": llm_i["response"],
                            "score_large": llm_i["fine-grained_score"],
                            "model_small": slm_i["model"],
                            "response_small": slm_i["response"],
                            "score_small": slm_i["fine-grained_score"],
                        }
                    )
    path_dataset = "datasets_nuf/"
    os.makedirs(path_dataset, exist_ok=True)

    df_out = pd.DataFrame(res)
    # df_out.to_pickle(path_dataset + "nuf_orig.pkl")
    df_out.to_pickle(path_dataset + "uflm2_orig.pkl")
    print("saved model: ", df_out.shape)

