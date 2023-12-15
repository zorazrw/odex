"""General argument initialization. """

import os 
import argparse
from src.utils import get_test_path, get_prediction_dir


class Arguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self.initialize_parser()
    
    def initialize_parser(self): 
        self.parser.add_argument("--model_name", type=str, default="codegen", 
            choices=["codegen", "codex"])
        self.parser.add_argument("--model_size", type=str, default="350M", 
            choices=["350M", "2B", "6B", "16B"], help="Size of avaiable models.")
        self.parser.add_argument("--model_data", type=str, default="mono", 
            choices=["nl", "multi", "mono"], help="(Last) training dataset applied.")
        self.parser.add_argument("--model_path", type=str, default=None)

        self.parser.add_argument("--per_gpu_batch_size", type=int, default=1)
        
        self.parser.add_argument("--local_rank", type=int, default=-1, 
            help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1, 
            help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument("--seed", type=int, default=0, 
            help="Random seed for initialization.")
        self.parser.add_argument("--dtype", type=str, default=None,
            choices=["int8", "float16"])
    
    def add_eval_args(self): 
        # dataset 
        self.parser.add_argument("--language", type=str, default="en", 
            choices=["en", "es", "ja", "ru"], help="Language subset for evaluation.")
        self.parser.add_argument("--input_path", type=str, default=None,
            help="Path of annotated datasets.")
        self.parser.add_argument("--output_dir", type=str, default=None,
            help="Directory to write the model predictions.")
        self.parser.add_argument("--output_name", type=str, default=None,
            help="Filename of the output model predictions.")
        self.parser.add_argument("--num_tests_input", type=int, default=0, 
            help="Number of annotated test cases included in the prompt.")
        self.parser.add_argument("--num_tests_eval", type=int, default=100, 
            help="Upperbound number of annotated test cases used for verfication.")
        self.parser.add_argument("--num_examples", type=int, default=0, 
            help="Number of examples included in the current prompt input. ")
        self.parser.add_argument("--fewshot_method", type=str, default="random", 
            choices=["random", "short"], 
            help="Method to select the prefix examples for prompt creation.")
        self.parser.add_argument("--function_name", type=str, default="id", 
            choices=["id", "constant", "intent"], 
            help="Method to construct the function name. ")
        self.parser.add_argument("--strip_prompt", action="store_true",
            help="Whether to strip the trailing whitespaces in the prompt. ")
        
        # generation
        self.parser.add_argument("--max_length_input", type=int, default=512, 
            help="Maximum number of tokens allowed in model inputs.")
        self.parser.add_argument("--max_length", type=int, default=512, 
            help="Maximum number of tokens allowed in model predictions.")
        self.parser.add_argument("--num_beams", type=int, default=10, 
            help="Beam size used for search. Has to be larger than `num_return_sequences.`")
        self.parser.add_argument("--num_return_sequences", type=int, default=10, 
            help="Number of predictions to return for each inference pass.")
        self.parser.add_argument("--temperature", type=float, default=0.8, 
            choices=[0.2, 0.6, 0.8], help="Temperature values attempted in the CodeGen paper.")
        self.parser.add_argument("--top_p", type=float, default=0.95, 
            help="Default top-p value used in the CodeGen paper.")
        
        # report 
        self.parser.add_argument("--eval_print_freq", type=int, default=100,
            help="Frequency of report printing during evaluation.")
        self.parser.add_argument("--verbose", action="store_true", 
            help="If printing intermediate i/o in a verbose mode. ")
    
    def parse(self):
        args = self.parser.parse_args()

        if args.input_path is None: 
            args.input_path = get_test_path(args.language)
            print(f"Input Path: {args.input_path}")

        if args.output_dir is None: 
            args.output_dir = get_prediction_dir(args.model_name)
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(
                os.path.join(args.output_dir, f"{args.language}-{args.model_size}-{args.model_data}-predictions"), 
                exist_ok=True
            )
        if args.output_name is None: 
            args.output_name = f"{args.language}-{args.model_size}-{args.model_data}-pred.jsonl"
        args.output_path = os.path.join(args.output_dir, args.output_name)

        args.model_name = f"Salesforce/codegen-{args.model_size}-{args.model_data}"
        if args.model_path is not None: 
            args.model_name_or_path = args.model_path 
        else: 
            args.model_name_or_path = args.model_name

        if hasattr(args, "num_return_sequences"): 
            if (not hasattr(args, "num_beams")) or (args.num_beams < args.num_return_sequences): 
                args.num_beams = args.num_return_sequences
        
        return args