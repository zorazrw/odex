"""CodeBLEU Metric. 

Including 4 components: 
1. N-gram match 
2. Weighted n-gram match
3. Syntax match
4. Dataflow match
"""

import os
import re 
import argparse 
import evaluate
from typing import List 

from . import (
    bleu, weighted_ngram_match, 
    syntax_match, dataflow_match, 
)
from .nmt_bleu import compute_bleu
from nltk.translate.bleu_score import corpus_bleu

# load prediction/reference files 
def load_single_file(path: str) -> List[str]: 
    snippets = [
        line.strip() for line in 
        open(path, 'r', encoding='utf-8').readlines()
    ]
    return snippets   # [num-samples]

def load_ref_files(path_list: List[str]) -> List[List[str]]: 
    file_references = [
        load_single_file(path) for path in path_list
    ]
    file_lengths = [len(r) for r in file_references]
    assert min(file_lengths) == max(file_lengths)

    references = [] 
    for idx in range(file_lengths[0]): 
        idx_ref_list = [r[idx] for r in file_references]
        references.append(idx_ref_list)
    return references   # [num-samples, num-references]

def tokenize_for_bleu_eval(code: str) -> List[str]:
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    if not tokens: tokens.extend(["",""])  # len(hyp) > 1 or bleu zero-division error  
    return tokens


# calculate n-gram match (bleu)
def calc_ngram_match(
    tokenized_hyps: List[List[str]], 
    tokenized_refs: List[List[List[str]]], 
    bleu_option: str, 
) -> float: 
    bleu_eval_metric = evaluate.load("bleu")
    if bleu_option == "hf": 
        hf_bleu_score = bleu_eval_metric.compute(
            predictions=tokenized_hyps, 
            references=tokenized_refs, 
            max_order=4, smooth=True, 
        )
        return hf_bleu_score["bleu"]
    elif bleu_option == "nmt": 
        nmt_bleu_scores = compute_bleu(
            reference_corpus=tokenized_refs, 
            translation_corpus=tokenized_hyps, 
            max_order=4, 
            smooth=True, 
        )
        return nmt_bleu_scores[0]
    elif bleu_option == "nltk": 
        nltk_bleu_score = corpus_bleu(tokenized_refs,tokenized_hyps)
        return nltk_bleu_score
    else: # "codebleu"
        ngram_match_score = bleu.corpus_bleu(
            hypotheses=tokenized_hyps, 
            list_of_references=tokenized_refs, 
        )
        return ngram_match_score


# calculate weighted n-gram match
def load_language_keywords(language: str) -> List[str]: 
    keyword_path = os.path.join("metric", "codebleu", "keywords", f"{language}.txt")
    keywords = [
        word.strip() for word in 
        open(keyword_path, 'r', encoding='utf-8').readlines()
    ]
    return keywords

def calc_weighted_ngram_match(
    tokenized_hyps: List[List[str]], 
    tokenized_refs: List[List[List[str]]], 
    language: str, 
) -> float: 
    keywords = load_language_keywords(language)

    def make_weights(reference_tokens: List[str], keyword_list: List[str]): 
        return {
            token:1 if token in keyword_list else 0.2 
            for token in reference_tokens
        }
    
    tokenized_refs_with_weights = [
        [
            [ref_tokens, make_weights(ref_tokens, keywords)] 
            for ref_tokens in reference
        ] 
        for reference in tokenized_refs
    ]
    weighted_ngram_match_score = weighted_ngram_match.corpus_bleu(
        tokenized_refs_with_weights,tokenized_hyps
    )
    return weighted_ngram_match_score


# calculate syntax match 
def calc_syntax_match(
    references: List[List[str]], 
    hypotheses: List[str], 
    language: str, 
) -> float: 
    return syntax_match.corpus_syntax_match(
        references=references, 
        candidates=hypotheses, 
        lang=language, 
    )


# calculate dataflow match 
def calc_dataflow_match(
    references: List[List[str]], 
    hypotheses: List[str], 
    language: str, 
) -> float: 
    return dataflow_match.corpus_dataflow_match(
        references=references, 
        candidates=hypotheses, 
        lang=language, 
    )



def compute_codebleu(
    predictions: List[str], 
    references: List[str], 
    language: str = "python", 
    params: List[float] = [0.25, 0.25, 0.25, 0.25], 
    bleu_option: str = "codebleu", 
    verbose: bool = False, 
) -> float: 
    tokenized_predictions = [
        tokenize_for_bleu_eval(sample_pred) 
        for sample_pred in predictions
    ]  # List[List[str]]
    tokenized_references = [
        [tokenize_for_bleu_eval(sref) for sref in sample_ref_list] 
        for sample_ref_list in references
    ]  # List[List[List[str]]]

    ngram_match_score = calc_ngram_match(
        tokenized_hyps=tokenized_predictions, 
        tokenized_refs=tokenized_references, 
        bleu_option=bleu_option, 
    )
    weighted_match_score = calc_weighted_ngram_match(
        tokenized_hyps=tokenized_predictions, 
        tokenized_refs=tokenized_references, 
        language=language, 
    )
    syntax_match_score = calc_syntax_match(
        references=references, 
        hypotheses=predictions,
        language=language, 
    )
    dataflow_match_score = calc_dataflow_match(
        references=references, 
        hypotheses=predictions,
        language=language, 
    )
    if verbose: 
        print(
            f"[N-gram Match  ] {ngram_match_score:.5f}\n"
            f"[Weighted Match] {weighted_match_score:.5f}\n"
            f"[Syntax Match  ] {syntax_match_score:.5f}\n"
            f"[Dataflow Match] {dataflow_match_score:.5f}\n"
        )

    codebleu_score = sum([
        weight * score for weight, score in 
        zip(params, [
            ngram_match_score, weighted_match_score, 
            syntax_match_score, dataflow_match_score
        ])
    ])
    if verbose: 
        print(f"CodeBLEU Score: {codebleu_score:.4f}")
    return codebleu_score



# main 
def main(): 
    hypotheses = load_single_file(args.hyp)
    references = load_ref_files(args.refs)

    codebleu_score = compute_codebleu(
        predictions=hypotheses, 
        references=references, 
        language=args.lang, 
        params=args.params, 
        bleu_option=args.bleu_option, 
        verbose=args.verbose, 
    )
    print(f"[CodeBLEU] {codebleu_score:.4f}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", type=str, nargs='+', 
        required=True, help="Reference files, one reference per line. ")
    parser.add_argument("--hyp", type=str, 
        required=True, help="Hypothesis fille, one prediction per line. ")
    parser.add_argument("--lang", type=str, default="python", 
        choices=["java", "js", "c_sharp", "php", "go", "python", "ruby"])
    parser.add_argument("--params", type=float, nargs='+', 
        default=[0.25, 0.25, 0.25, 0.25], 
        help="Weights for each evaluation component. ")
    
    parser.add_argument("--bleu_option", type=str, default="codebleu", 
        choices=["hf", "nmt", "nltk", "codebleu"])
    parser.add_argument("--verbose", action="store_true", 
        help="If printing intermediate and ultimate calculation results.")

    args = parser.parse_args()

    main()