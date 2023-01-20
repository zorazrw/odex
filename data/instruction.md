# Instruction

## Why Do This?

CoNaLa/MCoNaLa are dataset benchmarks to generate Python code snippets from natural language inputs written in English, Spanish, Japanese, and Russian. For example,

```
[intent] check if all elements in a list are identical
[rewritten_intent] check if all elements in list `myList` are identical
[snippet] all(x == myList[0] for x in myList)
[question_id] 3844801
```

By inputting the `rewritten_intent`, we expect models to generate code like the `snippet`.
Our goal is to evaluate the __functional correctness__ of a piece of generated code, by __writing unit tests__ for these samples (basically to provide enough variable and result specifications) and execute the prediction to verify their functional correctness.

## How to Write Unit Tests for an (M)CoNaLa Example?

### 1: Identify the variable names and string literals

In the `rewritten_intent` field, variable names are specified with ‘ \` ’ and string literals should be quoted with ‘ ‘’/ “” ’ (however sometimes they may falsely use ‘ ` ’). Identify which are adjustable variable names and which are constant values. For example (en-14),

```
[intent] How to join mixed list (array) (with integers in it) in Python?
[rewritten_intent] concatenate elements of list `b` by a colon ":"
[snippet] """:""".join(str(x) for x in b)
[question_id] 13954222
```

`b` is the input variable, but colon “:” is a constant.

### 2: Wrap a function outside the snippet

To execute a code snippet with enough context specification, we need to wrap it into a function with necessary arguments (part of the specified variable names). Following the above example, given the model prediction “b.concatenate([‘:’])”, we can write it as:

```
def func(b):
    return b.concatenate([':'])
```

Note that, not all of the variables should be the input arguments, since sometimes an argument needs to be returned rather than inputted, such as (en-21):

```
[intent] How can I convert a string with dot and comma into a float number in Python
[rewritten_intent] convert a string `my_string` with dot and comman into a float numer `my_float`
[snippet] my_float = float(my_string.replace(',', ''))
[question_id] 6633523

def func(my_string):
    my_float = float(my_string.replace(',', ''))
    return my_float
```

For these cases, we also need to write a suffix to fully wrap the snippet.

### 3: Wrap the ground truth snippet as well (Optional, but strongly recommended)

Refer to the jupyter notebook for more annotation details.
Note that, sometimes the given reference snippet cannot be correctly executed, please modify it (or rewrite one by referring to the original StackOverflow post) to create a valid `canonical_solution`.

### 4: Write unit tests and verify via execution

Import necessary libraries (numpy, pandas, etc.). Assign specific values to the input arguments and specify the desired output. Take the previous example (en-21), test with an assertion:
```assert func('1,234.00') == 1234.0```

Then run the code-eval-metric, and check if each snippet works:

```
{
    {'pass@1': 0.5, 'pass@2': 1.0}, 
    defaultdict(list,
        {0: [0, {'completion_id': 0, 'passed': False, 'result': 'failed: ', 'task_id': 0}]}, 
        {1: [1, {'completion_id': 1, 'passed': True, 'result': 'passed', 'task_id': 0}]}
    )
}
```

### _Notes_

1. Sometimes the intent looks ambiguous or under-specified. You can also refer to the original post with the corresponding `quesrion_id`, by specifying an url: (1) in English, use f“stackoverflow.com/q/{question-id}”, and (2) in lang=”ja”/”ru”/”es”, use f”{lang}.stackoverflow.com/q/{question-id}”.
2. For non-English MCoNaLa samples, you might want to use Google Translate or DeepL to better understand the natural language intent.

Please take a more detailed look at the notebook!
