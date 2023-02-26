"""Code-specific BLEU score. 
Use smoothing and 4-gram by default. 

 - `predictions`: one prediction (str) per sample, each should be tokenized into tokens (List[str])
 - `references`: multiple references (List[str]) per sample, each reference be tokenized (List[List[str]])
"""

import evaluate
from typing import List 
from metric.utils import tokenize_for_bleu_eval


def compute_bleu(predictions: List[str], references: List[str]) -> float: 
    """Compute bleu metric. """
    bleu_eval_metric = evaluate.load("bleu")
    predictions = [
        tokenize_for_bleu_eval(sample_pred) 
        for sample_pred in predictions
    ]
    references = [
        [tokenize_for_bleu_eval(sample_ref)] 
        for sample_ref in references
    ]
    bleu_scores = bleu_eval_metric.compute(
        predictions=predictions, 
        references=references, 
        max_order=4, smooth=True, 
    )
    return bleu_scores["bleu"]