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


from typing import List 
from metric.utils import tokenize_for_bleu_eval

from datasets import load_metric
meteor_eval_metric = load_metric("meteor")


def compute_meteor(predictions: List[str], references: List[str]) -> float: 
    """Compute meteor scores. """
    predictions = [
        ' '.join(tokenize_for_bleu_eval(sample_pred))
        for sample_pred in predictions
    ]
    references = [
        ' '.join(tokenize_for_bleu_eval(sample_ref)) 
        for sample_ref in references
    ]
    meteor_scores = meteor_eval_metric.compute(
        predictions=predictions, 
        references=references,
    )
    return meteor_scores["meteor"]