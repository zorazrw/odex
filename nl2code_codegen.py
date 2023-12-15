"""Evaluating CodeGen Performance on NL-to-Code Generation. 

No wrapped prompt for CodeGen, just comment-type nl descriptions. 
E.g., "# this function prints hello world"

CodeGen condictions on the concatenation of interleaved 
past prompts (nl) and generated responses (code). 
We can input the `test_start` as previous-step code. 
However, we cannot inform model the `suffix` (return arguments) beforehand,
hopefully the variable specification in the intent could help.
"""

import gc
import json
import torch
import src.slurm, src.config, src.data, src.utils

from typing import Dict, List  
from pathlib import Path
from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch


TRUC_PATTERN_LIST = [] # [r"\n\n^#", "^'''"]  # removed "\n\n\n"

def print_scores(scores_dict: Dict) -> str: 
    return f"{scores_dict}"

def remove_input_from_outputs(predictions: List[str], prompt: str, verbose: bool = False) -> List[str]: 
    # prompt_sections = [f"def {p}" for p in prompt.split("def ") if p]
    # for pp in prompt_sections: print(f"Sub Prompt: {pp}")
    if verbose: 
        print(f"Prompt: \n{prompt}")

    trimmed_predictions = [] 
    print(f"Loaded {len(predictions)} Original Predictions")
    for pred in predictions: 
        if prompt in pred: 
            s = pred.index(prompt)
            e = s + len(prompt)
            # print(f"Original Pred: {pred}")
            trimmed_pred = pred[: s] + pred[e: ]
            if "\n\ndef" in trimmed_pred: 
                trimmed_pred = trimmed_pred[: trimmed_pred.index("\n\ndef")]
            if trimmed_pred.startswith("return "): 
                trimmed_pred = trimmed_pred[len("return "): ]
            if verbose: 
                print(f"Trimmed Pred: \n{trimmed_pred}")
        else:
            trimmed_pred = pred
        trimmed_predictions.append(trimmed_pred)
    print(f"Collected {len(trimmed_predictions)} (trimmed) predictions")
    return trimmed_predictions


def evaluate(model, dataloader, tokenizer, args): 
    model.eval()
    if hasattr(model, "module"): 
        model = model.module

    gen_kwargs = {
        "max_length": args.max_length, 
        "num_beams": args.num_beams, 
        "num_return_sequences": args.num_return_sequences, 
        "temperature": args.temperature, 
        "top_p": args.top_p, 
    }

    total = 0

    write_path = Path(args.output_dir) / f"{args.language}-{args.model_size}-{args.model_data}-predictions"
    fw = open(write_path / (f"{args.global_rank}.jsonl"), 'a')
    print(f"Create sub-file: {write_path / (f'{args.global_rank}.jsonl')}")

    with torch.no_grad():
        for i, batch_inputs in enumerate(dataloader): 
            
            batch_prompts = batch_inputs["prompt"]
            batch_inputs = {
                k:v.to(model.device) for k,v in batch_inputs.items()
                if k != "prompt"
            }
            
            outputs = model.generate(**batch_inputs, **gen_kwargs)

            s, e = 0, gen_kwargs["num_return_sequences"]
            batch_size = batch_inputs["input_ids"].size(0)
            for j in range(batch_size): 
                j_preds = tokenizer.batch_decode(
                    outputs[s: e], 
                    skip_special_tokens=True, 
                    truncate_before_pattern=TRUC_PATTERN_LIST, 
                )
                j_prompt = batch_prompts[j]
                j_preds = remove_input_from_outputs(j_preds, j_prompt, args.verbose)

                j_dict = {"predictions": j_preds}
                fw.write(json.dumps(j_dict) + '\n')

                s += gen_kwargs["num_return_sequences"]
                e += gen_kwargs["num_return_sequences"]
                
                total += 1 
            
            if (i + 1) % args.eval_print_freq == 0: 
                log = f"Process rank: {args.global_rank}, {i+1} / {len(dataloader)}"
                logger.warning(log)
            
    
    logger.warning(f"Process rank:{args.global_rank}, total {total} ")

    if args.is_distributed:
        torch.distributed.barrier()




def main(): 
    torch.cuda.empty_cache()
    gc.collect()

    model_kwargs = {}
    if args.world_size > 1: 
        model_kwargs["device_map"] = "balanced_low_0"
    if args.dtype is not None: 
        if args.dtype == "int8":
            model_kwargs["load_in_8bit"] = True
        else: 
            model_kwargs["torch_dtype"] = torch.bfloat16 # else torch.float16 
    print(f"[Model Kwargs] {model_kwargs}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token  # '50256' use eos as pad token

    eval_examples = src.data.load_data(
        path=args.input_path, 
        global_rank=args.global_rank, 
        world_size=args.world_size, 
    )
    eval_dataset = src.data.Dataset(
        data=eval_examples, 
        num_tests=args.num_tests_input, 
        num_examples=args.num_examples, 
        fewshot_method=args.fewshot_method,
        function_name=args.function_name,
        strip_prompt=args.strip_prompt,
    )
    eval_sampler = SequentialSampler(eval_dataset)

    tokenization_kwargs = {
        "max_length": args.max_length_input, 
        "truncation": True, 
        "padding": True, 
        "return_tensors": "pt", 
    }
    collator_function = src.data.Collator(tokenizer, **tokenization_kwargs)
    eval_dataloader = DataLoader(
        eval_dataset, 
        sampler=eval_sampler, 
        batch_size=args.per_gpu_batch_size, 
        num_workers=args.world_size,
        collate_fn=collator_function,
    )

    # with init_empty_weights():
    #     config = AutoConfig.from_pretrained(args.model_name_or_path)
    #     model = AutoModelForCausalLM.from_config(config)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    logger.info(f"Model Using Device: {model.device}")
    if args.world_size <= 1: 
        model = model.to(args.device)
    logger.info(f"Model Using Device: {model.device}")
    
    logger.info("Start eval")
    scores_dict = evaluate(model, eval_dataloader, tokenizer, args)
    logger.info(f"Scores: {scores_dict}")

    if args.is_main:
        glob_path = Path(args.output_dir) / f"{args.language}-{args.model_size}-{args.model_data}-predictions"
        write_path = args.output_path
        src.utils.write_output(glob_path, write_path) 



if __name__ == "__main__": 
    parser = src.config.Arguments()
    parser.add_eval_args()
    args = parser.parse()

    src.slurm.init_distributed_mode(args)
    src.slurm.init_signal_handler()

    if args.is_distributed: 
        torch.distributed.barrier()
    logger = src.utils.init_logger(
        args.is_main, args.is_distributed, 
        Path(args.output_dir) / "run.log"
    )

    if not Path(args.output_dir).exists() and args.is_main:
        parser.print_options(args)

    main()