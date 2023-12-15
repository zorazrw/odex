"""Evaluate Codex performance on NL-to-Code generation. """

import os, random  
import time, json, argparse
from typing import Dict, List  
from src.utils import get_test_path, get_prediction_path, load_testset
from prompt import create_fewshot_prompt_nl2code
from verify import get_valid_solutions, wrap_check

import openai 
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_response_001(
    prompt: str, 
    verbose: bool = False, 
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    response = openai.Completion.create(
        model=args.model_name, 
        prompt=prompt, 
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        n=args.n, 
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"], 
    )
    return [choice["text"] for choice in response["choices"]]


def get_response_davinci_002(
    prompt: str, 
    sample: Dict, 
    verbose: bool = False, 
) -> List[str]: 
    if verbose: print(f"[prompt] \n{prompt}\n------")

    response = openai.Completion.create(
        model=args.model_name, 
        prompt=prompt, 
        suffix=sample["suffix"], 
        max_tokens=args.max_tokens, 
        temperature=args.temperature, 
        top_p=args.top_p, 
        n=args.n, 
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["###"], 
    )
    return [choice["text"] for choice in response["choices"]]

RESPOND_DICT = {
    "code-cushman-001": get_response_001, 
    "code-davinci-001": get_response_001, 
    "code-davinci-002": get_response_davinci_002,
}

def get_predictions(
    model_name: str, prompt: str, 
    sample: Dict, index: int, 
    sleep_time: int, verbose: bool, 
) -> List[str]:
    if sleep_time == 0: 
        return RESPOND_DICT[model_name](prompt, sample, verbose)
    
    # enable sleep time otherwise
    predictions = None 
    while predictions is None: 
        try: 
            predictions = RESPOND_DICT[model_name](prompt, sample, verbose)
        except: 
            print(f"sleep for {sleep_time} at sample #{index}")
            time.sleep(sleep_time)
    return predictions


def select_fewshot_examples(
    sample: Dict, candidates: List[Dict], 
    num_examples: int = 1, method: str = "random", 
) -> List[Dict]: 
    """Select example as prefix to the prompt of the current sample. """
    if method == "random": 
        num_examples = min(num_examples, len(candidates))
        return random.sample(candidates, num_examples)
    



def main(): 
    # load source dataset
    dataset = load_testset(args.input_path)
    
    predset = [] 
    scores_dict = {f"pass@{idx}": [] for idx in range(1, args.n+1)}

    for i, sample in enumerate(dataset): 
        # create model input -- prompt 
        examples = select_fewshot_examples(
            sample=sample, 
            candidates=dataset[:i]+dataset[i+1:], 
            num_examples=args.num_examples, 
            method=args.fewshot_method,
        )
        prompt = create_fewshot_prompt_nl2code(
            sample=sample, 
            examples=examples, 
            num_tests=args.num_tests, 
            function_name=args.function_name, 
        )
        if args.strip_prompt:
            prompt = prompt.rstrip()

        # collect code predictions 
        predictions = get_predictions(
            model_name=args.model_name, 
            prompt=prompt, sample=sample, index=i, 
            sleep_time=args.sleep_time, verbose=args.verbose, 
        )
        
        # simple cleansing of predicions
        valid_predictions = get_valid_solutions(predictions, deduplicate=False)
        num_valid = len(valid_predictions)
        assert num_valid == args.n, f"# num_valid"
        scores, outputs = wrap_check(
            sample, valid_predictions, 
            k=[i+1 for i in range(num_valid)], 
            num_workers=args.n, 
            max_num_tests=args.num_tests_eval, 
            verbose=args.verbose, 
            function_name=args.function_name, 
        )
        if i % 10 == 0: 
            print(f"[scores@{i:3d}] {scores}")

        for idx in range(num_valid): 
            key = f"pass@{idx+1}"
            if key in scores: 
                scores_dict[key].append(scores[key])
        predset.append({
            "output": outputs, 
            "predictions": valid_predictions, 
        })
    
    # write records to prediction file 
    json.dump(predset, open(args.output_path, 'w'))
    
    for idx in range(args.n): 
        key = f"pass@{idx+1}"
        scores = scores_dict[key]
        print(f"[{key}] {sum(scores)/len(scores):.3f} ({len(scores)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)

    parser.add_argument("--num_tests", type=int, default=0)
    parser.add_argument("--num_tests_eval", type=int, default=100)

    parser.add_argument("--model_name", type=str, default="code-davinci-002", 
        choices=["code-cushman-001", "code-davinci-001", "code-davinci-002"])
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=10, 
        help="Number of predictions required for each api call.")
    
    parser.add_argument("--sleep_time", type=int, default=60, 
        help="Specify a positive integer if enable time sleep.")
    
    parser.add_argument("--function_name", type=str, default="id", 
        choices=["id", "constant", "intent"], 
        help="Method to construct the function name. ")
    parser.add_argument("--num_examples", type=int, default=0, 
        help="Number of examples included in the current prompt input. ")
    parser.add_argument("--fewshot_method", type=str, default="random", 
        choices=["random"], 
        help="Method to select the prefix examples for prompt creation.")
    parser.add_argument("--strip_prompt", action="store_true",
        help="Whether to strip the trailing whitespaces in the prompt. ")

    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if (not args.input_path) or (not args.output_path): 
        if not args.language: 
            raise Exception(f"Need to specify [language] or [i/o path]")
        if not args.input_path: 
            args.input_path = get_test_path(args.language)
        if not args.output_path: 
            args.output_path = get_prediction_path(
                args.model_name, args.language, 
                args.num_examples, args.num_tests, 
            )

    if args.openai_api_key is not None: 
        openai.api_key = args.openai_api_key


    main()