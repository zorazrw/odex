"""Dataset utilities. 
"""

import json 
import torch 
import random
from typing import Dict, List 
from prompt import create_fewshot_prompt_nl2code 


def load_data(path, global_rank: int = -1, world_size: int = -1) -> List[Dict]: 
    assert path and path.endswith(".jsonl")

    examples = []
    fr = open(path, 'r')
    for idx, example in enumerate(fr): 
        if (global_rank > -1) and (idx%world_size != global_rank): 
            continue
        example = json.loads(example)
        examples.append(example)

    fr.close()
    print(f"Loaded #{len(examples)} data samples onto [{global_rank}].")
    return examples


class Dataset(torch.utils.data.Dataset): 
    def __init__(
        self, data: List[Dict], num_tests: int = 0, 
        num_examples: int = 0, fewshot_method: str = "random", 
        function_name: str = "id", strip_prompt: bool = False,
    ): 
        self.data = data
        self.num_tests = num_tests
        self.num_examples = num_examples
        self.fewshot_method = fewshot_method
        self.function_name = function_name
        self.strip_prompt = strip_prompt

        if self.fewshot_method == "short": 
            indexed_data = [(i, s) for i,s in enumerate(self.data)]
            indexed_data = sorted(indexed_data, key=lambda x: len(x[1]["canonical_solution"]))
            self.candidate_indices = [i for i,s in enumerate(indexed_data[:100])]
        else:
            self.candidate_indices = [i for i in range(len(self.data))]
    
    def __len__(self): 
        return len(self.data)
    
    def get_example(self, index: int): 
        return self.data[index]
    
    def get_target(self, example: Dict) -> str:
        return example["canonical_solution"]
    
    def get_prefix_examples(self, exclude_indices: List[int]): 
        candidate_indices = [i for i in self.candidate_indices if i not in exclude_indices]
        num_examples = min(self.num_examples, len(candidate_indices))
        selected_indices = random.sample(candidate_indices, num_examples)
        return [self.data[idx] for idx in selected_indices]
    
    def __getitem__(self, index): 
        sample = self.get_example(index)
        assert sample["intent"] is not None
        examples = self.get_prefix_examples(exclude_indices=[index])
        prompt = create_fewshot_prompt_nl2code(
            sample=sample, 
            examples=examples, 
            num_tests=self.num_tests, 
            function_name=self.function_name, 
        )
        if self.strip_prompt:
            prompt = prompt.strip()
        return {
            "prompt": prompt, 
        }


class Collator(object): 
    def __init__(self, tokenizer, **kwargs): 
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        print(f"[Collator Kwargs]: {self.kwargs}")
    
    def __call__(self, batch: List[Dict]) -> Dict: 
        prompts = [ex["prompt"] for ex in batch]
        inputs_dict = self.tokenizer(
            prompts, **self.kwargs, 
        ) # {"input_ids": -, "attention_mask": - }
        inputs_dict["prompt"] = prompts
        return inputs_dict