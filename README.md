
# Overview

This repository is an extension of the code for the paper [Task Ambiguity in Humans and Language Models](https://arxiv.org/abs/2212.10711).

It focuses on generating datasets of AmbiBench, a new benchmark of six ambiguously-specified classification tasks. The goal of AmbiBench is to construct a testbed of minimal complexity where we can control and measure the degree of ambiguity in various task specifications.

The code contains functionality to test language models on the three different AmbiBench settings discussed in the paper:
1.  task disambiguation using natural language instructions
2.  task disambiguation using multiple examples
3.  finetuning a model to generalize well in the face of ambiguity

# Setup

1.  create a virtualenv (https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

2.  ``pip install -r requirements.txt``


# Generate AmbiBench dataset

When calling _main.py_, you can add arguments specifying:
```python
  --construction_types {subject_location,propn_negation,religious_pronoun,location,subject,negation,pronoun,religious,propn} [{subject_location,propn_negation,religious_pronoun,location,subject,negation,pronoun,religious,propn} ...]
                        Provide 1 or more tasks/categories for which to generate
                        examples
  --construction_format {arrow,qa}
  --n_shots N_SHOTS     Number of shots per query
  --n_queries N_QUERIES
                        Number of queries/examples to generate
  --n_multiple_choices N_MULTIPLE_CHOICES
                        Number of multiple-choice categories
  --output_dir OUTPUT_DIR
                        Directory where to store dataset as JSON
  --needs_instruction
  --needs_informative
  --needs_multiple_choice
  --include_ambiguous_examples
  --no_salient_task     Do not explicitly use construction type as salient task.
  --verbose VERBOSE
  --prob_of_ambiguous PROB_OF_AMBIGUOUS
  --seed SEED           Random seed
```


## 1.  Task disambiguation using natural language instruction
Example command:
``main.py --shots=20 --need_informative=False``

For the arguments for the argparse defined in _main.py_, make sure that ``shots = 20``, ``need_informative = False``.

## 2.  Task disambiguation using multiple examples
Example command:
``main.py --shots=1 --need_informative=False``

Make sure that ``shots = 1``, ``need_informative = True`` if running test with informative instructions and ``False`` if running test with uninformative instructions, and model is set to whatever model you want to test on.

## 3.  Finetuning a model to generalize well in the face of ambiguity
Example command:
``main.py --shots=20 --need_informative=False``

Make sure that ``shots = 20``, ``need_informative = False``.

If running the control experiments (finetuning on unambiguous data), set ``finetuning_control = True``. If running the ambiguous experiments, set ``finetuning_control = False``.

# Visualization
e.g:

``v = Visualizer(all_tests, args.needs_instruction)``
``v.visualize_accuracy()``

Create a new Visualizer object and call the function corresponding to the test you ran (docstrings for each function available in ``visualizer.py``). Generally, for (1), use ``visualize_accuracy``. And for (2), use ``visualize_accuracy_across_shots``. And for (3), use ``plot_individual_finetuning_performance_for_heldout``.


In ``visualizer.py`` you can set the output path for the generated figure in the last line of each function.

# Documenting important bits of Code

## `Prompt` class
```
Creates a prompt for the OpenAI API using by generating examples
    A Prompt consists of three Examples (the last one being called the query), metadata on each of those Examples, and in some cases, an instruction
    For example, a Prompt may look like:
        Instruction
        Example 1 {metadata}
        Example 2 {metadata}
        Query {metadata}
```

To generate examples, it determines the corresponding generator type given a `construction_type = {SubjectLocation; PropnNegation; ReligiousPronoun}`.

In the simple (i.e. non-salient) case,

# How to generate a set of examples and prompts?

To create a set examples, we wish to obtain a JSON file with queries and their expected completions. The examples should be constructed based on the configuration provided by arguments (e.g., `needs_informative = True`).

```
{
    "date": "YY-MM-DD_HH-mm",
    "configuration": {
        "arg1": "val1"
    },
    "examples": [
        {
            "query": "query text",
            "completion": "X"
        },
        {
            "query": "another query text",
            "completion": "Y"
        },
    ]

}

```


# How to determine salient category label?

The salient category is the specific category that underlies the labels for a set of examples. It is more specific than the salient task - which takes values like `subject` or `location` (stored in `ConstructionType`).

First, the salient task needs to be determined. This is necessary because `Examples` are generated randomly, so we need to infer the salient task that would have produced the `Example`.
