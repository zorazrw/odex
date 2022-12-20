"""ROUGE score. 
Compute rouge-1/2/L ?

 - `predictions`: one prediction (str) per sample -> List[str]
 - `references`: one reference (str) per sample -> List[str]
"""

from typing import List 
from metric.utils import tokenize_for_bleu_eval

from datasets import load_metric
rouge_eval_metric = load_metric("rouge")


def compute_rouge(predictions: List[str], references: List[str]) -> float: 
    """Compute rouge scores. """
    predictions = [
        ' '.join(tokenize_for_bleu_eval(sample_pred))
        for sample_pred in predictions
    ]
    references = [
        ' '.join(tokenize_for_bleu_eval(sample_ref)) 
        for sample_ref in references
    ]
    rouge_scores = rouge_eval_metric.compute(
        predictions=predictions, 
        references=references,
    )
    return rouge_scores["rougeL"].mid.fmeasure
