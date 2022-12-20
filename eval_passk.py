"""Evaluate nl2code performance using model predictions. 
- using different numbers of test cases for evaluation
"""

import json, argparse, itertools
import numpy as np 
from typing import Dict, List 
from verify import wrap_check 
from src.utils import (
    get_test_path, get_prediction_path, 
    load_testset, print_scores_dict, 
)


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([
        estimator(int(n), int(c), k) 
        for n, c in zip(num_samples_it, num_correct)
    ])

def re_score(predset: List[str], n: int) -> Dict: 
    total, correct = [], []
    for i, plist in enumerate(predset): 
        if isinstance(plist, dict): plist = plist["output"]
        assert len(plist) == n, f"length {len(plist)}"
        passed = [int(output["passed"]) for _, output in plist]

        total.append(len(passed))
        correct.append(sum(passed))
        
    total, correct = np.array(total), np.array(correct)
    scores_dict = {
        f"pass@{k+1}": estimate_pass_at_k(total, correct, k+1).tolist() 
        for k in range(args.k) if (total >= k).all()
    }
    return scores_dict

def remove_comment_lines(code: str) -> str: 
    code_lines = code.split('\n')
    code_lines = [cl for cl in code_lines if not cl.lstrip().startswith("#")]
    unindented_code_lines = []
    for cl in code_lines: 
        if cl.startswith('\t'): 
            unindented_code_lines.append(cl[1:])
        elif cl.startswith("    "):
            unindented_code_lines.append(cl[4:])
        else:
            unindented_code_lines.append(cl)
    return '\n'.join(unindented_code_lines)

def re_evaluate(testset: List[Dict], predset: List[Dict]) -> Dict: 
    scores_dict = {f"pass@{idx+1}": [] for idx in range(args.k)}

    for idx, (sample, predict) in enumerate(zip(testset, predset)): 
        pred_list = predict["predictions"][: args.k]
        assert len(pred_list) == args.k, f"got prediction list with #{len(pred_list)} samples"

        scores, outputs = wrap_check(
            sample, pred_list, 
            k=[i+1 for i in range(args.k)], 
            num_workers=args.num_workers, 
            max_num_tests=args.num_tests_eval, 
            verbose=args.verbose, 
            exclude_suffix=False, 
        )

        predset[idx]["outputs"] = outputs
        for ii in range(args.k): 
            key = f"pass@{ii+1}"
            scores_dict[key].append(scores[key])

        if (idx+1) % args.report_steps == 0: 
            print(f">>> step {idx+1} <<<")
            print_scores_dict(scores_dict, args.k)

    return scores_dict, predset


def re_eval_per_domain(testset: List[Dict], predset: List[Dict], output_path: str) -> Dict: 
    domain_scores_dict = {"none": []}

    for idx, (sample, predict) in enumerate(zip(testset, predset)): 
        pred_list = predict["predictions"][: args.k]
        
        scores, outputs = wrap_check(
            sample, pred_list, 
            k=[i+1 for i in range(args.k)], 
            num_workers=args.num_workers, 
            max_num_tests=args.num_tests_eval, 
            verbose=args.verbose, 
            exclude_suffix=args.exclude_suffix, 
        )
        if len(sample["library"]) == 0: 
            domain_scores_dict["none"].append(scores)
            print(f"pass@1: {scores['pass@1']}")
        for lib in sample["library"]: 
            if lib not in domain_scores_dict: 
                domain_scores_dict[lib] = []
            domain_scores_dict[lib].append(scores)
    
    aggr_scores_dict = {}
    for domain, scores_list in domain_scores_dict.items():
        freq = len(scores_list)
        aggr_scores = {}
        for i in range(args.k): 
            ikey = f"pass@{i+1}"
            i_scores = [scores[ikey] for scores in scores_list] 
            avg_i_scores = sum(i_scores) / len(i_scores)
            aggr_scores[ikey] = avg_i_scores

        aggr_scores_dict[domain] = {
            "count": freq, 
            "scores": aggr_scores, 
        }

    with open(output_path, 'w') as fw: 
        json.dump(aggr_scores_dict, fw)



def rewrite_predset(path: str, predset: List[Dict]): 
    if path.endswith(".json"): 
        with open(path, 'w') as fw: 
            json.dump(predset, fw)
    elif path.endswith(".jsonl"): 
        with open(path, 'w') as fw:
            for sample in predset:
                fw.write(json.dumps(sample) + '\n')
    else: 
        raise ValueError(f"Prediction path should be in JSON format, but got: {path}")


def load_predset(path: str) -> List[Dict]: 
    if path.endswith(".jsonl"): 
        predset = [json.loads(l.strip()) for l in open(path, 'r')]
    else: 
        predset = json.load(open(path, 'r'))
    return predset 

def main(): 
    testset = [] 
    for tpath in args.test_path: 
        sub_testset = load_testset(tpath)
        testset.extend(sub_testset)
    
    predset = []
    for ppath in args.prediction_path: 
        sub_predset = load_predset(ppath)
        predset.extend(sub_predset)


    # compare: library usage
    indices = [idx for idx in range(len(testset))]
    if args.library_usage == "closed": 
        indices = [idx for idx in indices if ("import " not in testset[idx]["test_start"])]
    elif args.library_usage == "open": 
        indices = [idx for idx in indices if ("import " in testset[idx]["test_start"])]
    
    # compare: number of input exemplar test cases
    if args.min_num_tests is not None: 
        indices = [idx for idx in indices if len(testset[idx]["test"])>=args.min_num_tests]
    
    if args.indices: 
        indices = args.indices 
        
    testset = [testset[idx] for idx in indices]
    predset = [predset[idx] for idx in indices]

    print(f"<{args.library_usage}> samples #{len(testset)}")

    if args.do_reeval_domain: 
        re_eval_per_domain(testset, predset, args.domain_scores_path)
    else:
        if not args.no_re_eval:   # redo the evaluation, possibly with different test cases
            scores_dict, updated_predset = re_evaluate(testset, predset)
            if args.rewrite_predictions: 
                print(f"rewrite the latest outputs to prediction path")
                rewrite_predset(args.prediction_path, updated_predset)
        else:                 # load the recorded scores and rescore for a subset
            scores_dict = re_score(predset, args.k)
    
        print('\n\n')
        print(f"Overall Pass@K Scores: ")
        print_scores_dict(scores_dict, args.k)
        print('-'*30)

        open_indices = [idx for idx in indices if ("import " in testset[idx]["test_start"])]
        open_scores_dict = {k: [v[idx] for idx in open_indices] for k,v in scores_dict.items()}
        print(f"W/Lib Pass@K Scores: ")
        print_scores_dict(open_scores_dict, args.k)
        print('-'*30)

        closed_indices = [idx for idx in indices if ("import " not in testset[idx]["test_start"])]
        closed_scores_dict = {k:[v[idx] for idx in closed_indices] for k,v in scores_dict.items()}
        print(f"Closed Pass@K Scores: ")
        print_scores_dict(closed_scores_dict, args.k)
        print('-'*30)



if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--test_path", type=str, nargs='+', default=None)
    parser.add_argument("--prediction_path", nargs='+', type=str, default=None)

    parser.add_argument("--num_tests_eval", type=int, default=1)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--min_num_tests", type=int, default=None, 
        help="Filter samples with at least a certain number of test cases, if specified.")
    parser.add_argument("--library_usage", type=str, default="all", 
        choices=["all", "closed", "open"], 
        help="Filter samples with closed/open-domain operations. `all` by default.")

    parser.add_argument("--indices", type=int, nargs='+', default=[])
    parser.add_argument("--report_steps", type=int, default=50)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--exclude_suffix", action="store_true")

    parser.add_argument("--no_re_eval", action="store_true", 
        help="Performing re-scoring if not specified.")
    parser.add_argument("--rewrite_predictions", action="store_true", 
        help="Whether the rewrite the execution outputs, if `no_re_eval` not set.")
    
    parser.add_argument("--do_reeval_domain", action="store_true")
    parser.add_argument("--domain_scores_path", type=str, default="domain_scores.json")

    args = parser.parse_args()

    if (not args.test_path) or (not args.prediction_path): 
        if not args.language: 
            raise Exception(f"Need to specify [language] or [i/o path]")
        if not args.test_path: 
            args.test_path = get_test_path(args.language)
        if not args.prediction_path: 
            args.prediction_path = get_prediction_path(args.language, args.num_tests)
    
    main()