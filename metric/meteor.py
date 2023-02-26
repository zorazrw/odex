"""METEOR metric. 

Args:
    `predictions`: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    `references`: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    `alpha`: Parameter for controlling relative weights of precision and recall. default: 0.9
    `beta`: Parameter for controlling shape of penalty as a function of fragmentation. default: 3
    `gamma`: Relative weight assigned to fragmentation penalty. default: 0.5
Returns:
    'meteor': meteor score.
"""

import evaluate
from typing import List 
from metric.utils import tokenize_for_bleu_eval


def compute_meteor(predictions: List[str], references: List[str]) -> float: 
    """Compute meteor scores. """
    meteor_eval_metric = evaluate.load("meteor")
    predictions = [
        tokenize_for_bleu_eval(sample_pred)
        for sample_pred in predictions
    ]
    references = [
        tokenize_for_bleu_eval(sample_ref)
        for sample_ref in references
    ]
    meteor_scores = meteor_eval_metric.compute(
        predictions=predictions, 
        references=references,
    )
    return meteor_scores["meteor"]