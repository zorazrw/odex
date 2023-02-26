"""ChrF metric. 

Produces ChrF(++) scores for hypotheses given reference translations.

Args:
    predictions (list of str): The predicted sentences.
    references (list of list of str): The references. There should be one reference sub-list for each prediction sentence.
    char_order (int): Character n-gram order. Defaults to `6`.
    word_order (int): Word n-gram order. If equals to `2`, the metric is referred to as chrF++. Defaults to `0`.
    beta (int): Determine the importance of recall w.r.t precision. Defaults to `2`.
    lowercase (bool): if `True`, enables case-insensitivity. Defaults to `False`.
    whitespace (bool): If `True`, include whitespaces when extracting character n-grams.
    eps_smoothing (bool): If `True`, applies epsilon smoothing similar
    to reference chrF++.py, NLTK and Moses implementations. If `False`,
    it takes into account effective match order similar to sacreBLEU < 2.0.0. Defaults to `False`.

Returns:
    'score' (float): The chrF (chrF++) score,
    'char_order' (int): The character n-gram order,
    'word_order' (int): The word n-gram order. If equals to 2, the metric is referred to as chrF++,
    'beta' (int): Determine the importance of recall w.r.t precision
"""

import evaluate
from typing import List 
from metric.utils import tokenize_for_bleu_eval


def compute_chrf(predictions: List[str], references: List[str]) -> float: 
    """Compute ChrF scores. """
    chrf_eval_metric = evaluate.load("chrf")
    predictions = [
        tokenize_for_bleu_eval(sample_pred)
        for sample_pred in predictions
    ]
    references = [
        [tokenize_for_bleu_eval(sample_ref)] 
        for sample_ref in references
    ]
    chrf_scores = chrf_eval_metric.compute(
        predictions=predictions, 
        references=references,
    )
    return chrf_scores["score"]