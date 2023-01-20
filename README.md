# Execution-based Evaluation for Open Domain Code Generation

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This repository contains the data and code for the work [Execution-based Evaluation for Open Domain Code Generation](https://arxiv.org/pdf/2212.10481.pdf).

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

If you find our paper or code useful, please cite the paper

```
@article{wang2022execution,
  title={Execution-Based Evaluation for Open-Domain Code Generation},
  author={Zhiruo Wang, Shuyan Zhou, Daniel Fried, Graham Neubig},
  journal={arXiv preprint arXiv:2212.10481},
  year={2022}
}
```

## Install

```bash
pip install -r requirements.txt
```

## Dataset

We split the dataset by which natural language is of the corresponding intent.

```markdown
.
├── README.md
├── data
│   ├── en_test.jsonl
│   ├── es_test.jsonl
│   ├── ja_test.jsonl
│   └── ru_test.jsonl
```

Each line contains a serialized json object, an example looks like:

```
{
    'task_id': 3844801,
    'intent': "check if all elements in list `myList` are identical", 
    'prompt': "def f_3844801(myList):\n\treturn ",
    'canonical_solution': "all(x == myList[0] for x in myList)",
    'suffix': "",
    'test_start': "\ndef check(candidate):",
    'test': [
        "\n    assert candidate([1,2,3]) == False\n", 
        "\n    assert candidate([1,1,1,1,1,1]) == True\n",
        "\n    assert candidate([1]) == True\n",
        "\n    assert candidate(['k','k','k','k','k']) == True\n",
        "\n    assert candidate([None,'%$#ga',3]) == False\n"
    ],
    'entry_point': "f_3844801",
}
```

where:

1. `task_id` is the post id of the original StackOverflow post where the sample is constructed from;
2. `intent` is the natural language description rewritten by human annotators with qualified specificity;
3. `prompt` is the function prefix (definition, input arguments, etc.) to properly execute the code snippet;
4. `canonical_solution` is the reference solution (verified by human annotators) of the coding problem;
5. `suffix` is the function suffix (return values, if any) to proper to execute the code;
6. `test_start` is the definition of test functions, also, including library imports if necessitated by the program;
7. `test` is the list of test cases created by human annotators;
8. `entry_point` is the function name that should be called for 'check' during evaluation.

To correctly execute the (canonical) code snippets, one needs to install all involved libraries, as listed in the `./library/` directory.


## Evaluating Code Generation Models

We provide code to evaluate on two state-of-the-art code generation models: CodeX and CodeGen. To perform the NL-to-Code generation task and collect model predictions:

For __CodeX__, run

```bash
python nl2code_codex.py --language en \
--model_name "code-davinci-002" \
--openai_api_key ${YOUR_API_KEY} \
```

change the `model_name` argument to "code-cushman-001" or "code-davinci-001" to try other model variants.

For __CodeGen__, run

```bash
python nl2code_codegen.py --language en \
--model_size 350M --model_data mono 
```

Other valid options for `model_size` include: "2B", "6B", and "16B", which correspond to the 2.7B, 6.1B, and 16.1B CodeGen models.

For `model_data`, other options include "multi" and "nl".

### Evaluation

Our default evaluation metric is the execution pass rate.
Before the evaluation, make sure your environment has all required libraries installed, and better imported as in the code samples. To do this, you can:

```bash
pip install -r ./library/requirements.txt 
python ./library/imports.py
```

Then we can perform the execution by running:

```bash
python eval_passk.py --language en --prediction_path ${MODEL_PRED_PATH}
```

We also support five other non-execution metrics: BLEU, ROUGE, METEOR, ChrF, and CodeBLEU.
For example, to evaluate with the BLEU metric, run:

```bash
python eval_nonex.py --language en --prediction_path ${MODEL_PRED_PATH} --eval_metric bleu
```

Specifying the `eval_metric` argument with "rouge"/"meteor"/"chrf"/"codebleu" to use other metrics.

### Detailed Analysis

#### Open-Domain versus Closed-Domain

To evaluate on the subset of open-domain or closed-domain samples, you only need to add another argument at evaluation time (when running `eval_passk.py` or `eval_nonex.py`), by

```bash
--library_usage "open"   # or "closed"
```

#### Few-shot Prompting

To include more prompt-solution pairs for in-context prompting learning, specify the `num_examples` at inference time (when running `nl2code_codex.py` and `nl2code_codegen.py`), by

```bash
--num_examples 1    # 2, 3, ... 
```

#### Number of Input Test Cases

To add exemplar test cases in the prompt inputs, specify the `num_tests` at inference time, by

```bash
--num_tests 1   # 2, 3, ...
```

#### Number of Evaluation Test Cases

To use different numbers of test cases for execution-based evaluation, specify the `num_tests_eval` when running `eval_passk.py`, for example

```bash
python eval_passk.py --language en --prediction_path ${MODEL_PRED_PATH} --num_tests_eval 1 
```

#### Semantics of Function Names

Our paper explores three methods to create function names in the wrapping context:

* "id": `f_${ID}`, simple string formatting using the StackOverflow post ID
* "constant": `function`, using the same string constant for all samples
* "intent": heuristic-based extraction from the paired NL intent

To experiment with different function names, specify the `function_name` at inference time, by

```bash
--function_name "intent"   # "id" "constant"
```

#### Metric Correlation

We also provide code to compare execution-based and non-execution evaluation metrics on a sample-wise basis. Take the execution and BLEU score as an example, one can run:

```bash
python metric_corr.py --language en \
--prediction_file ${MODEL_PRED_PATH} \
--eval_metric "bleu"
```

To get visualizations in violin plots and histograms, add `--do_plot_violin` or `do_plot_stacked_hist`.
