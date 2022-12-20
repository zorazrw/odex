"""Evaluate Model Predictions on Non-Execution based Metrics."""

import json 
import argparse
from typing import Dict, List 
from src.utils import get_test_path, get_prediction_path, load_testset
from metric import (
    compute_bleu, compute_rouge, compute_meteor, 
    compute_chrf, compute_codebleu,
)

MetricDict = {
    "bleu": compute_bleu, 
    "rouge": compute_rouge, 
    "meteor": compute_meteor, 
    "chrf": compute_chrf, 
    "codebleu": compute_codebleu, 
}

def calc_sample_score(evaluator, sample: Dict, predict: List[str], top_k: int = 1) -> float: 
    top_k = min(len(predict), top_k)
    scores = [
        evaluator(predictions=[predict[i]], references=[sample["canonical_solution"]])
        for i in range(top_k)
    ]
    return max(scores)

def calc_corpus_score(evaluator, dataset: List[Dict], predset: List[Dict], index: int) -> float: 
    predictions = [p["predictions"][index] for p in predset]
    references = [s["canonical_solution"] for s in dataset]
    score = evaluator(
        predictions=predictions, 
        references=references, 
    )
    return score

def main():
    testset = load_testset(args.test_path)
    if args.prediction_path.endswith(".jsonl"): 
        predset = [json.loads(l.strip()) for l in open(args.prediction_path, 'r')]
    else:
        predset = json.load(open(args.prediction_path, 'r'))

    indices = [idx for idx in range(len(testset))]
    if args.library_usage == "closed": 
        indices = [idx for idx in indices if ("import " not in testset[idx]["test_start"])]
    elif args.library_usage == "open": 
        indices = [idx for idx in indices if ("import " in testset[idx]["test_start"])]
    
    if args.indices: 
        indices = args.indices 
        
    testset = [testset[idx] for idx in indices]
    predset = [predset[idx] for idx in indices]

    print(f"<{args.library_usage}> samples #{len(testset)}")

    evaluator = MetricDict[args.eval_metric]

    if args.average_type == "micro":  # bleu
        scores = [] 
        for i in range(args.top_k):
            score = calc_corpus_score(evaluator, testset, predset, i)
            scores.append(score)
        print(f"Scores: {scores}")
        avg_score = max(scores)
    elif args.average_type == "macro":
        scores = [] 
        for sample, predict in zip(testset, predset): 
            if isinstance(predict, dict): 
                score = calc_sample_score(evaluator, sample, predict["predictions"], args.top_k)
            else: 
                score = calc_sample_score(evaluator, sample, predict, args.top_k)
            scores.append(score)
        avg_score = sum(scores) / len(scores)
    
    print(f"Metric [{args.eval_metric}]: {avg_score:.4f}")




if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--prediction_path", type=str, default=None)

    parser.add_argument("--library_usage", type=str, default="all", 
        choices=["all", "closed", "open"], 
        help="Filter samples with closed/open-domain operations. `all` by default.")
    parser.add_argument("--indices", type=int, nargs='+', default=[])

    parser.add_argument("--eval_metric", type=str, default="bleu", 
        choices=["bleu", "rouge", "meteor", "chrf", "codebleu"])
    parser.add_argument("--top_k", type=int, default=10, 
        help="Taking best scores from top-k predictions.")
    parser.add_argument("--average_type", type=str, default="micro", 
        choices=["micro", "macro"])
    
    args = parser.parse_args()

    if (not args.test_path) or (not args.prediction_path): 
        if not args.language: 
            raise Exception(f"Need to specify [language] or [i/o path]")
        if not args.test_path: 
            args.test_path = get_test_path(args.language)
        if not args.prediction_path: 
            args.prediction_path = get_prediction_path(args.language, args.num_tests)
    
    if args.eval_metric == "codebleu": 
        if args.average_type != "macro": 
            print(f"Set average method for metric [{args.eval_metric}] to [macro]")
            args.average_type = "macro"
    else: 
        if args.average_type != "micro": 
            print(f"Set average method for metric [{args.eval_metric}] to [micro]")
            args.average_type = "micro"
    
    main()