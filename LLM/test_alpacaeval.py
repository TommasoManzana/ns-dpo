import argparse
import torch
import json
import os
import random
import warnings
import numpy as np
import time
import transformers

import datasets
# from datasets import load_dataset
from vllm import LLM, SamplingParams
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm.lora.request import LoRARequest

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

parser = argparse.ArgumentParser()
parser.add_argument("--config_path1", type=str, default="./config/model/llama3.2-1b.yaml")
parser.add_argument("--original_model", type=str, required=True)
parser.add_argument("--name_model", type=str, default="ufbf-p1.0-cp81-nsdpo-g0.85-b0.1", required=True)
parser.add_argument("--path_model", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--seed", type=int, default=2024)

args = parser.parse_args()

def tokenize_batch_chat_template(tokenizer, examples, device):

    assert tokenizer.padding_side == 'left'

    outputs = list()
    for i in range(len(examples["instruction"])):
        out = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": examples["instruction"][i]}, 
                {"role": "assistant", "content": "None"}
            ],
            tokenize=False, add_generate_prompt=True
        ).split("None")[0]
        # out = tokenizer.encode(
        #     examples["instruction"][i], 
        #     add_special_tokens=True,
        #     # return_tensors="pt",
        # )[0]

        outputs.append(out)
    return outputs

    # # find max length:
    # max_len = max([len(out) for out in outputs])

    # # pad all outputs to max length:
    # padded_outputs = list()
    # attention_masks = list()
    # for out in outputs:
    #     if len(out) < max_len:
    #         len_out = len(out)
    #         out = torch.concat([torch.tensor([tokenizer.pad_token_id]*(max_len-len(out))), out])
    #         padded_outputs.append(out)

    #         attention_mask = torch.concat([torch.zeros(max_len-len_out), torch.ones(len_out)])
    #         attention_masks.append(attention_mask)

    #     else:
    #         padded_outputs.append(out)
    #         attention_masks.append(torch.ones(len(out)))
            
    # return {'input_ids': torch.stack(padded_outputs).to(device), 'attention_mask':torch.stack(attention_masks).to(device)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path_llm = args.path_model
name_file = args.name_model
# max_examples = 64 # Set this to a positive value to test on smaller number of prompts
max_examples = -1 # Set this to a positive value to test on smaller number of prompts

USE_LORA = False
path_savedir = "./results_alpacaeval/"
model_dir = path_savedir + name_file
os.makedirs(model_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    args.original_model,
    padding_side="left"
)
tokenizer.pad_token = tokenizer.eos_token

try:
    if path_llm != "none":
        llm = LLM(
          model=model_dir, 
          tokenizer=args.original_model,
          tensor_parallel_size=1,
        )
    else:
        llm = LLM(
          model=args.original_model,
          tokenizer=args.original_model,
          tensor_parallel_size=1,
        )
    print(f"Model at {model_dir} successfully loaded")
except:
    model = AutoModelForCausalLM.from_pretrained(
        args.original_model
    ).to(device)
    path_pt = path_llm + "/LATEST/policy.pt"
    model.load_state_dict(torch.load(path_pt)["state"])
    model.save_pretrained(model_dir)
    print(f"VLLM model saved at {model_dir}")

    llm = LLM(
      model=model_dir, 
      tokenizer=args.original_model,
      tensor_parallel_size=1,
    )
    print(f"Newly created VLLM model saved at {model_dir} loaded")

# Load evaluation dataset for AlpacaEval
eval_set = datasets.load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]

# Generate evaluation responses from the model
res = list()
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    seed=args.seed,
    max_tokens=2048,
    #max_tokens=64, # set it to higher value like 2048 for proper test
)

time_start = time.time()

idx_examples = 0
continue_generate = True

while continue_generate:
    end_idx = min(len(eval_set), idx_examples + args.batch_size)
    examples = eval_set[idx_examples:end_idx]
    batch_templates = tokenize_batch_chat_template(tokenizer, examples, device)

    examples["prompt_template"] = batch_templates
    
    outputs = llm.generate(
        # examples["instruction"],
        examples["prompt_template"],
        sampling_params
    )

    examples["output"] = [output.outputs[0].text for output in outputs]
    examples["generator"] = [name_file] * len(examples["instruction"])
    
    res += [{k: examples[k][i] for k in examples} for i in range(len(examples["instruction"]))]

    idx_examples += end_idx - idx_examples
    if max_examples > 0 and idx_examples >= max_examples:
        break
    elif idx_examples >= len(eval_set):
        break
        
        

time_elapsed = time.time() - time_start
print(f"{time_elapsed / 3600:.2f} hours passed to finish generating {idx_examples} responses")

# Save generated responses

with open(f"{path_savedir}/{name_file}.json", "w") as f:
    json.dump(res, f)

# Set OpenAI API key with 'export OPENAI_API_KEY=YOUR_KEY'.
# Run $ alpaca_eval --model_outputs 'test_alpacaeval/responses.json'.
    
