"""Utility Functions. 
 - code specific tokenization
"""

import re 
from typing import List 


def tokenize_for_bleu_eval(code: str, return_tokens: bool = False) -> List[str]:
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]
    if not tokens: tokens.extend(["",""])  # len(hyp) > 1 or bleu zero-division error  
    if return_tokens: return tokens
    return ' '.join(tokens)