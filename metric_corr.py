"""Correlation between Execution-based Metrics and BLEU Scores. 
 - stacked histogram 
 - violin plot 

Potentially relevant evaluation metrics from huggingface/evaluate repository
 - BLEU:     ["bleu", "google_bleu", "sacrebleu"]
 - ROUGE:    ["rouge"]
 - METEOR:   ["meteor"]
 - ChrF:     ["chrf"]
 - CodeBLEU: ["idsedykh/codebleu", "idsedykh/codebleu2"]
 - RUBY:     []
 - others:   ["code_eval", "exact_match", "f1", "kaggle/ai4code"]
"""

import json
import argparse
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 
from typing import Dict, List, Tuple 

from src.utils import load_testset
from metric import (
    compute_bleu, compute_rouge, compute_meteor, 
    compute_chrf, compute_codebleu,
)


def bucket_scores(none_scores: List[List[float]], exec_scores: List[List[float]]) -> Dict: 
    pass_none_scores, fail_none_scores = [], [] 
    for sample_none_scores, sample_exec_scores in zip(none_scores, exec_scores): 
        for b, e in zip(sample_none_scores, sample_exec_scores): 
            if e == 0.0: fail_none_scores.append(b)
            else: pass_none_scores.append(b)
    
    print(f"#pass: {len(pass_none_scores)}, #fail: {len(fail_none_scores)}")
    return {
        "pass": {
            "bleu": np.array(pass_none_scores, dtype=float), 
            "exec": np.array([1. for _ in range(len(pass_none_scores))], dtype=float),
        }, 
        "fail": {
            "bleu": np.array(fail_none_scores, dtype=float), 
            "exec": np.array([0. for _ in range(len(fail_none_scores))], dtype=float),
        }, 
    }

def auto_figure_name(predict_path: str, plot_type: str) -> str: 
    return predict_path.replace(".json", f"_{plot_type}.png")


def plot_violin(scores_dict: Dict, scale: float = 1.0):
    pass_none_scores = scores_dict["pass"]["bleu"]
    num_pass = len(pass_none_scores)
    fail_none_scores = scores_dict["fail"]["bleu"]
    num_fail = len(fail_none_scores)

    plt.ylim(0.0, scale * 1.01)
    plt.violinplot([pass_none_scores, fail_none_scores])
    plt.xticks([1.0, 2.0], ["pass", "fail"])
    plt.ylabel("none score", fontstyle="italic")

    pass_fail_info = f"(pass: {num_pass} / fail: {num_fail})"
    if args.plot_title: plt.title(args.plot_title + ' ' + pass_fail_info)

    figname = auto_figure_name(args.prediction_file, f"violin-{args.eval_metric}")
    plt.savefig(figname, format='png', dpi=500)
    plt.clf()


def plot_stacked_hist(
    scores_dict: Dict, 
    colors: List[str] = ["skyblue", "gold"], 
    bins: int = 20, linewidth: float = 2.5, 
    edgecolor: str = "white", alpha: float = 0.5, 
    scale: float = 1.0, 
): 
    pass_bleu_scores = scores_dict["pass"]["bleu"]
    num_pass = len(pass_bleu_scores)
    print(pass_bleu_scores[:10])
    fail_bleu_scores = scores_dict["fail"]["bleu"]
    num_fail = len(fail_bleu_scores)
    print(fail_bleu_scores[:10])

    s, e = 0.0 * scale, 1.01 * scale
    bin_sticks = np.arange(s, e, scale/bins)

    plt.hist(
        x=[fail_bleu_scores], color=colors[0], label="fail", 
        bins=bin_sticks, linewidth=linewidth, edgecolor=edgecolor,alpha=alpha, 
    )
    plt.hist(
        x=[pass_bleu_scores], color=colors[1], label="pass", 
        bins=bin_sticks, linewidth=linewidth, edgecolor=edgecolor,alpha=alpha, 
    )
    plt.xlabel(f"{args.eval_metric} score", fontstyle="italic")
    plt.ylabel("frequency", fontstyle="italic")
    plt.legend(loc="upper center")
    
    pass_fail_info = f"(pass: {num_pass} / fail: {num_fail})"
    if args.plot_title: plt.title(args.plot_title + ' ' + pass_fail_info)
    
    figname = auto_figure_name(args.prediction_file, f"hist-{args.eval_metric}")
    plt.savefig(figname, format='png', dpi=500)
    plt.clf()



def tf_bucket(bleu_scores: List[List[float]], exec_scores: List[List[float]]): 
    correct_bleu_scores, false_bleu_scores = bucket_scores(bleu_scores, exec_scores)
    
    plt.hist(
        x=[correct_bleu_scores, false_bleu_scores], 
        color=["yellowgreen", "salmon"], 
        label=["correct", "false"], 
        bins=args.n_bins, 
        linewidth=args.linewidth, 
        edgecolor="white", 
    )
    plt.xlabel("bleu score", loc="center")
    plt.ylabel("frequency", loc="center")
    plt.xlim([0, 1])
    plt.ylim([0, args.max_freq])
    if args.plot_title: plt.title(args.plot_title)
    plt.legend()
    # plt.show()
    figname = args.prediction_file.replace(".json", ".png")
    plt.savefig(figname, format='png', dpi=500)


def dist_ttest(bleu_scores: List[List[float]], exec_scores: List[List[float]]): 
    """Test if distributions of two groups of bleu scores are statistically different. 
    Reference Docs: https://www.geeksforgeeks.org/how-to-conduct-a-two-sample-t-test-in-python/
    """
    correct_bleu_scores, false_bleu_scores = bucket_scores(bleu_scores, exec_scores)
    correct_bleu_scores = np.array(correct_bleu_scores)
    false_bleu_scores = np.array(false_bleu_scores)
    print(f"[correct] mean: {np.mean(correct_bleu_scores):.4f}, var: {np.var(correct_bleu_scores):.4f}")
    print(f"[false]   mean: {np.mean(false_bleu_scores):.4f}, var: {np.var(false_bleu_scores):.4f}")

    scipy_result = stats.ttest_ind(a=correct_bleu_scores, b=false_bleu_scores, equal_var=True)
    print(f"[scipy-ttest] \n{scipy_result}")

    pg_result = pg.ttest(
        x=correct_bleu_scores, 
        y=false_bleu_scores, 
    )
    print(f"[pg-ttest] \n{pg_result}")


def flatten_scores(bleu_scores: List[List[float]], exec_scores: List[List[float]]) -> Tuple[List[float], List[float]]: 
    return (
        [b for scores in bleu_scores for b in scores], 
        [e for scores in exec_scores for e in scores]
    )


def plot_scatter(bleu_scores: List[List[float]], exec_scores: List[List[float]]): 
    pass_bleu_scores, fail_bleu_scores = bucket_scores(bleu_scores, exec_scores)
    pass_exec_scores = [1. for _ in range(len(pass_bleu_scores))]
    fail_exec_scores = [0. for _ in range(len(fail_bleu_scores))]

    pass_bleu_scores = np.array(pass_bleu_scores, dtype=float)
    pass_exec_scores = np.array(pass_exec_scores, dtype=float)

    fail_bleu_scores = np.array(fail_bleu_scores, dtype=float)
    fail_exec_scores = np.array(fail_exec_scores, dtype=float)

    plt.scatter(pass_bleu_scores, pass_exec_scores, color="skyblue", s=3)
    plt.scatter(fail_bleu_scores, fail_exec_scores, color="goldenrod", s=3)

    plt.xlabel("BLEU score", fontsize=10, fontstyle="italic")
    plt.xlabel("PASS score", fontsize=10, fontstyle="italic")
    plt.show()



MetricDict = {
    "bleu": compute_bleu, 
    "rouge": compute_rouge, 
    "meteor": compute_meteor, 
    "chrf": compute_chrf, 
    "codebleu": compute_codebleu, 
}

ScaleDict = {
    "bleu": 1.0, 
    "rouge": 1.0, 
    "meteor": 1.0, 
    "chrf": 100.0, 
    "codebleu": 1.0, 
}

def main():
    # load annotated and prediction samples 
    testset = load_testset(args.test_path)
    predset = json.load(open(args.prediction_file, 'r'))
    assert len(testset) == len(predset)
    print(f"#Dataset [{len(testset)}]; #Predset [{len(predset)}]")

    # collect non- and executable scores
    # E.g., bleu - exec
    none_scores, exec_scores = [], []

    none_evaluator = MetricDict[args.eval_metric]

    for idx, (sample, predict) in enumerate(zip(testset, predset)): 
        # non-execution based score
        sample_none_scores = [
            none_evaluator(
                predictions=[p], 
                references=[sample["canonical_solution"]], 
            )
            for p in predict["predictions"]
        ]
        none_scores.append(sample_none_scores)

        # execution correctness
        sample_exec_scores = [float(o["passed"]) for i,o in predict["output"]]
        exec_scores.append(sample_exec_scores)

        if (idx + 1) % args.report_steps == 0: print(f"#{idx + 1}")
    
    scores_dict = bucket_scores(none_scores, exec_scores)
    metric_scale = ScaleDict[args.eval_metric]

    if args.do_plot_stacked_hist:
        plot_stacked_hist(
            scores_dict, colors=args.colors, 
            bins=args.bins, linewidth=args.linewidth, 
            scale=metric_scale, 
        )
    if args.do_plot_violin:
        plot_violin(scores_dict, scale=metric_scale)
    
    if args.do_tf_bucket:
        tf_bucket(none_scores, exec_scores)
    if args.do_dist_ttest: 
        dist_ttest(none_scores, exec_scores)
    if args.do_plot_scatter: 
        plot_scatter(none_scores, exec_scores)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en", 
        choices=["en", "es", "ja", "ru"])
    parser.add_argument("--test_path", type=str, required=True, 
        help="File of the original annotated test cases.")
    parser.add_argument("--prediction_file", type=str, required=True, 
        help="File containing test cases only.")
    
    parser.add_argument("--eval_metric", type=str, default="bleu", 
        choices=["bleu", "rouge", "meteor", "chrf", "codebleu"],
        help="To Implement: code-bert-score, ruby!!")
    
    # violin plot
    parser.add_argument("--do_plot_violin", action="store_true")

    # histogram plot 
    parser.add_argument("--do_plot_stacked_hist", action="store_true")
    parser.add_argument("--colors", type=str, nargs='+', 
        default=["skyblue", "gold"])
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--linewidth", type=float, default=2.5)

    # specified title 
    parser.add_argument("--plot_title", type=str, default=None)
    

    parser.add_argument("--do_tf_bucket", action="store_true")
    
    parser.add_argument("--max_freq", type=int, default=1200)
    

    parser.add_argument("--do_dist_ttest", action="store_true")

    parser.add_argument("--do_plot_scatter", action="store_true")
    
    # report freq
    parser.add_argument("--report_steps", type=int, default=50)

    args = parser.parse_args()
    
    # if args.do_plot_violin or args.do_plot_stacked_hist: 
    #     args.plot_title = f"[{args.language.upper()}]"
    main()