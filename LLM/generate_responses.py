import argparse
import os, sys
import torch
import torch.nn as nn
import transformers

import pickle
import pandas as pd
import numpy as np
from time import time
from src.datasets.rms import *
from src.datasets.non_stationary_datasets import remove_time_and_var_from_prompt
from transformers import BitsAndBytesConfig, pipeline, set_seed

from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer

def tokenize_batch_chat_template(tokenizer, messages, device):

    assert tokenizer.padding_side == 'left'

    outputs = list()
    for message in messages:
        # out = tokenizer.encode(
        #     message["content"], 
        #     add_special_tokens=True,
        #     return_tensors="pt",
        # )[0]
        out = tokenizer.apply_chat_template(
            [message], 
            add_special_tokens=True,
            return_tensors="pt",
            add_generate_prompt=True
        )[0]

        outputs.append(out)

    # find max length:
    max_len = max([len(out) for out in outputs])

    # pad all outputs to max length:
    padded_outputs = list()
    attention_masks = list()
    for out in outputs:
        if len(out) < max_len:
            len_out = len(out)
            out = torch.concat([torch.tensor([tokenizer.pad_token_id]*(max_len-len(out))), out])
            padded_outputs.append(out)

            attention_mask = torch.concat([torch.zeros(max_len-len_out), torch.ones(len_out)])
            attention_masks.append(attention_mask)

        else:
            padded_outputs.append(out)
            attention_masks.append(torch.ones(len(out)))
            
    return {'input_ids': torch.stack(padded_outputs).to(device), 'attention_mask':torch.stack(attention_masks).to(device)}

def evaluate_KL(args, model, ref_model, tokenizer, inputs, num_samples=32):

    res = list()
    for i in range(inputs["input_ids"].shape[0]):
        target = inputs["input_ids"][i]
        outputs = model.generate(
            input_ids=target.unsqueeze(0),
            attention_mask=inputs["attention_mask"][i].unsqueeze(0),
            max_new_tokens=32,
            num_return_sequences=num_samples,
            do_sample=True,  # Enable sampling
            top_k=50,        # Sampling method (optional)
            temperature=1.0,  # Sampling temperature (optional)
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Calculate log probabilities of the generated sequences
        log_probs = []
        for output in outputs:
            with torch.no_grad():
                # output = output[inputs["input_ids"][i].shape[-1]:]
                # Get the logits for the entire sequence
                logits = model(output.unsqueeze(0)).logits  # Shape: (1, seq_len, vocab_size)
                logits_ref = ref_model(output.unsqueeze(0)).logits  # Shape: (1, seq_len, vocab_size)
                # logits = model(output.unsqueeze(0)).logits[:, inputs["input_ids"][i].shape[-1]:, :]  # Shape: (1, seq_len, vocab_size)
                # logits_ref = ref_model(output.unsqueeze(0)).logits[:, inputs["input_ids"][i].shape[-1]:, :]  # Shape: (1, seq_len, vocab_size)

                j_logps = list()
                shifted_labels = output[1:]        # Ignore first token as label
                for vs in [logits, logits_ref]:
                    # Shift logits and labels to compute probabilities of each token
                    shifted_logits = vs[:, :-1, :]  # Ignore last token logits
                    
                    # Compute log probabilities
                    log_softmax = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
                    token_log_probs = log_softmax[0, torch.arange(shifted_labels.size(0)), shifted_labels]
                    # print(token_log_probs.shape, target.shape[-1])
                    
                    # Sum the log probabilities of the entire sequence
                    sequence_log_prob = token_log_probs[target.shape[-1]:].sum().item()
                    j_logps.append(sequence_log_prob)
                log_probs.append(j_logps[0] - j_logps[1])
        res.append(sum(log_probs) / len(log_probs)) # (inputs_len, num_samples)
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_model1", type=str, required=True)
    parser.add_argument("--original_model2", type=str, required=True)
    parser.add_argument("--model1_path", type=str, required=True)
    parser.add_argument("--model2_path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rmodel_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--test_length", type=int, default=-1)
    parser.add_argument("--name_model1", type=str, required=True)
    parser.add_argument("--name_model2", type=str, required=True)
    parser.add_argument("--ref_model", type=str, required=True)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_lora1", action='store_true', default=False)
    parser.add_argument("--use_lora2", action='store_true', default=False)
    parser.add_argument("--config_path1", type=str, default="./config/model/llama2-7b-chat-hf.yaml")
    parser.add_argument("--config_path2", type=str, default="./config/model/llama2-7b-chat-hf.yaml")
    args = parser.parse_args()

    time_start = time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    orig_models = [args.original_model1, args.original_model2]
    models = [args.model1_path, args.model2_path]
    names = [args.name_model1, args.name_model2]
    config_paths = [args.config_path1, args.config_path2]
    use_loras = [args.use_lora1, args.use_lora2]

    # load test data
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    prompts = list()
    # trim time step and var from prompts
    for key in data.keys():
        timestep = data[key]["timestep"]
        prompts.append(remove_time_and_var_from_prompt(key, timestep))

    if args.test_length > 0:
        prompts = prompts[:args.test_length]

    prompt_loader = torch.utils.data.DataLoader(prompts, batch_size=args.batch_size, shuffle=False)

    # load ref_model
    ref_model = transformers.AutoModelForCausalLM.from_pretrained(
        orig_models[0],
        # quantization_config=bnb_config,   
    ).to(device)
    ref_model.load_state_dict(torch.load(args.ref_model)["state"])

    df = {"prompts": prompts}
    l_kls = list()
    for i in range(2):
        l_kls.append(list())
        
        if use_loras[i]:
            
            # Load base model
            # compute_dtype = getattr(torch, dtype)
            
            base_model = transformers.AutoModelForCausalLM.from_pretrained(
                orig_models[i],
                quantization_config=bnb_config,   
            )
            
            # Load LoRA configuration
            config_lora = OmegaConf.load(config_paths[i])
            loraconfig = LoraConfig(
                r=config_lora.lora_rank,
                lora_alpha=config_lora.lora_alpha,
                target_modules=config_lora.lora_target_modules,
                # lora_dropout=config_lora.lora_dropout,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = PeftModel(base_model, loraconfig)
            # model = get_peft_model(base_model, loraconfig)
            
            # Load the state dict
            state_dict = torch.load(models[i])  # path to .pt file
            state_dict = state_dict['state']

            # inspect_lora_weights(model, state_dict, print_all=True)
            
            model.load_state_dict(state_dict)
            model = model.merge_and_unload()
            model.bfloat16()
            
        else:
            
            # load model
            if models[i] == "none":
                model = transformers.AutoModelForCausalLM.from_pretrained(
                    orig_models[i],
                    quantization_config=bnb_config,   
                )
            else:
                if "gpt2" in orig_models[i] or \
                    "Llama-3.2-1B" in orig_models[i] or \
                    "Qwen2" in orig_models[i]:
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        orig_models[i],
                    ).to(device)
                    model.load_state_dict(torch.load(models[i])["state"])
                else:
                    model = transformers.AutoModelForCausalLM.from_pretrained(
                        orig_models[i],
                    ).to(device)
                    model.load_state_dict(torch.load(models[i]))
                    
        model.eval()
        # breakpoint()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            orig_models[i],
            padding_side="left"
        )
        
        # if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        generator = None
        if "gpt2" in orig_models[i]:
            generator = pipeline('text-generation', model='gpt2-large')
        
        # generate responses
        responses = []
        num_generated = 0

        for prompt in prompt_loader:
            messages = [
                {"role": "user", "content": p}
                for p in prompt
            ]
    
            inputs = tokenize_batch_chat_template(tokenizer, messages, device)

            # generate outputs
            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=args.top_k,
            )
    
            for j in range(len(prompt)):
                response = tokenizer.decode(
                    outputs[j], 
                    skip_special_tokens=True
                )
                responses.append(response[len(prompt[j]):])

            # collect logps
            i_kls = evaluate_KL(args, model, ref_model, tokenizer, inputs)
            l_kls[-1] += i_kls
            
            num_generated += len(prompt)
            print(num_generated, " responses generated")
            if num_generated == args.test_length:
                break

        df[f"KL_model{i+1}"] = l_kls[-1]
        df[names[i]] = responses
        del model, tokenizer
        torch.cuda.empty_cache()

        # check and report GPU memory usage
        print(f"GPU memory usage after [{models[i]}]: {torch.cuda.memory_summary(device=None, abbreviated=False)}")
    
    df = pd.DataFrame(df)

    # load reward model
    if "pairrm" in args.rmodel_path.lower():
        rmodel, rtokenizer = load_pairrm("llm-blender/PairRM-hf")
    elif "armorm" in args.rmodel_path.lower():
        rmodel, rtokenizer = load_armorm()
    rmodel.cuda()
    
    # compute winrate
    win_model1 = list()
    win_model2 = list()
    total = 0

    if "pairrm" in args.rmodel_path.lower():
        prefs, logits, scores = apply_pairrm(rmodel, rtokenizer, df, "prompts", names[0], names[1])
    elif "armorm" in args.rmodel_path.lower():
        prefs, logits, scores = apply_armoRM(rmodel, rtokenizer, df, "prompts", names[0], names[1])

    df["win_model1"] = prefs
    df["win_model2"] = 1 - df["win_model1"]
    if scores is not None:
        df["score_model1"] = [score[0].item() for score in scores]
        df["score_model2"] = [score[1].item() for score in scores]
    
    winrate = sum(df["win_model1"]) / len(df["win_model1"])
    print(f"\nWinrate for {names[0]} vs {names[1]}: {winrate:.4f} vs {1 - winrate:.4f}")

    # evaluate KL-winrates
    df["KLwin_model1"] = (
        (df["score_model1"] - args.tau * df["KL_model1"]) > (df["score_model2"] - args.tau * df["KL_model2"])
    ) * 1.0 + (
        (df["score_model1"] - args.tau * df["KL_model1"]) == (df["score_model2"] - args.tau * df["KL_model2"])
    ) * 0.5

    # save results
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    df.to_csv(
        os.path.join(
            args.save_path, 
            f"{names[0]}-{names[1]}-{args.rmodel_path}-len{args.test_length}-seed{args.seed}.csv"
        ), 
        index=False
    )

    # append to existing summary file
    # new_summary = df.drop(columns=["prompts", "score_model1", "score_model2",  "KL_model1", "KL_model2", names[0], names[1]]).mean(axis=0)
    new_summary = df.drop(columns=["prompts", names[0], names[1]]).mean(axis=0)
    new_summary["model1"] = names[0]
    new_summary["model2"] = names[1]
    new_summary["rmodel"] = args.rmodel_path
    new_summary["length"] = args.test_length
    new_summary["seed"] = args.seed
    new_summary["max_tokens"] = args.max_tokens

    if os.path.exists(os.path.join(args.save_path, "summary.csv")):
        summary = pd.read_csv(os.path.join(args.save_path, "summary.csv"))
        summary = pd.concat(
            [summary, new_summary.to_frame().T], 
            ignore_index=True
        )
    else:
        summary = new_summary.to_frame().T

    summary.to_csv(os.path.join(args.save_path, "summary.csv"), index=False)

    print(f"Total time: {time() - time_start:.2f} seconds for {args.test_length} samples")

