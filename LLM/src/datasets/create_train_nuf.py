import os, sys
import re
import ast
import math
import json
import pickle
import datasets
import argparse
import transformers
import itertools
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple

sys.path.insert(0, "../../")
sys.path.insert(0, "../")

def ufb_process_into_dict(dataset:pd.DataFrame):

    #Create the output dictionary
    question_dict = dict()
    output_dict = defaultdict(list)

    #Convert the dataframe into a list of dict types    
    df_dict = dataset.dropna().replace({np.nan: None}).to_dict('records')

    not_appended = 0
    for datapoint in df_dict:
        
        try:
            #Set the question as the key and process the dict:
            question = datapoint['prompt'] + f"| Time step: {datapoint['timestep']} |"
            if question not in question_dict:
                question_dict[question] = 1
            else:
                question_dict[question] = question_dict[question] + 1
            question += f" STRIP THIS AWAY FOR VAR {question_dict[question]} STRIP THIS AWAY FOR VAR "
            output_dict[question] = {
                k: datapoint[k] for k in ["responses","pairs", "sft_target", "timestep"]
            }
        except:
            not_appended += 1
            # print('Unable to append time value to input prompt on datapoint:')
            # print(datapoint)

    print(f"not appended items: {not_appended} out of {len(df_dict)}")
    return output_dict

# small_lms = ['starchat', 'llama-2-7b-chat', 'mpt-30b-chat', 'wizardlm-7b', 'falcon-40b-instruct', 'pythia-12b', 'alpaca-7b',  'llama-2-13b-chat', 'wizardlm-13b', 'vicuna-33b', 'ultralm-13b']
small_lms = ['starchat', 'llama-2-7b-chat', 'wizardlm-7b', 'pythia-12b', 'alpaca-7b',  'llama-2-13b-chat', 'wizardlm-13b', 'ultralm-13b']
# large_lms = ['wizardlm-70b', 'llama-2-70b-chat', 'gpt-3.5-turbo', 'ultralm-65b', 'gpt-4', 'bard']
large_lms = ['llama-2-70b-chat', 'gpt-3.5-turbo', 'gpt-4', 'bard']

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--rhodiff", type=float, default=0.7)
    parser.add_argument("--cp", type=int, default=51)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_timesteps", type=int, default=100)
    parser.add_argument("--num_testdata", type=int, default=500)
    args = parser.parse_args()
    
    # df_orig = pd.read_pickle("datasets_nuf/nuf_orig.pkl").sample(frac=1, random_state=args.seed)
    df_orig = pd.read_pickle("datasets_nuf/uflm2_orig.pkl").sample(frac=1, random_state=args.seed)
    print("Data shapes: orig ", df_orig.shape)
    df_orig["responses"] = df_orig.apply(
        lambda row: [row["response_large"], row["response_small"]],
        axis=1
    )
    df_orig["pairs"] = [[(0, 1)]] * df_orig.shape[0]
    df_orig["sft_target"] = df_orig["response_large"]
    df_orig["timestep"] = args.num_timesteps + 1
    
    df_test = df_orig.iloc[-args.num_testdata:].copy()
    items_per_timestep = (df_orig.shape[0]-args.num_testdata) // 100
    df_train = df_orig.iloc[:-args.num_testdata].iloc[:items_per_timestep * 100].copy()
    df_train = df_train.reset_index(drop=True)
    print("Data shapes: train ", df_train.shape, ", test: ", df_test.shape)

    # change preferences based on the options
    assert args.rhodiff >= 0. and args.rhodiff <= 1.0, "rhodiff is out ot 0.0 <= rhodiff <= 1.0"
    assert args.cp <= args.num_timesteps, "cp should be <= num_timesteps"
    num_changed = int(df_train.shape[0] * args.rhodiff * (args.cp/args.num_timesteps))
    df_train.loc[:num_changed-1, "pairs"] = [[(1, 0)]] * num_changed
    df_train.loc[:num_changed-1, "sft_target"] = df_train.loc[:num_changed-1, "response_small"]
    df_train = df_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # assign time steps to df_train
    for i in range(args.num_timesteps):
        df_train.loc[items_per_timestep * i:items_per_timestep*(i+1)-1, "timestep"] = i + 1

    df_train = df_train.rename(columns={"instruction": "prompt"})
    df_test = df_test.rename(columns={"instruction": "prompt"})

    # breakpoint()
    
    dict_train = ufb_process_into_dict(
        df_train.drop(
            [
                "model_large",
                "response_large",
                "score_large",
                "model_small",
                "response_small",
                "score_small",
            ], 
            axis=1
        )
    )
    dict_test = ufb_process_into_dict(
        df_test.drop(
            [
                "model_large",
                "response_large",
                "score_large",
                "model_small",
                "response_small",
                "score_small",
            ], 
            axis=1
        )
    )

    # train_path = f"./datasets_nuf/nuf_train_t{args.num_timesteps}_rho{args.rhodiff}_cp{args.cp}.pkl"
    # test_path = f"./datasets_nuf/nuf_test_t{args.num_timesteps}_rho{args.rhodiff}_cp{args.cp}.pkl"
    train_path = f"./datasets_nuf/uflm2_train_t{args.num_timesteps}_rho{args.rhodiff}_cp{args.cp}.pkl"
    test_path = f"./datasets_nuf/uflm2_test_t{args.num_timesteps}_rho{args.rhodiff}_cp{args.cp}.pkl"
    
    with open(train_path, 'wb') as f:
        pickle.dump(dict_train, f)
    with open(test_path, 'wb') as f:
        pickle.dump(dict_test, f)
