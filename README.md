# CodeScoreEval

Graduation Thesis Project: A Large Language Model-Driven Approach for Automated Code Quality Evaluation


## Overview

CodeScoreEval is a research-oriented repository for building and evaluating an LLM-based system that assesses programming solutions across multiple dimensions of code quality. The project focuses on automatic scoring of code submissions for:

- Correctness
- Efficiency
- Readability

The repository combines prompt design, dataset processing, fine-tuning, inference, and evaluation into a single workflow suitable for academic experimentation and thesis development.

## Contributors

| Student ID | Name | GitHub |
|---|---|---|
| 22127460 | Quách Trần Quán Vinh | [@huhyhuvinh](https://github.com/huhyhuvinh) |
| 22127478 | Nguyễn Hoàng Trung Kiên | [@juinkinn](https://github.com/juinkinn) |

## Motivation

Traditional code evaluation often relies on handcrafted heuristics or limited human review. This project explores a more scalable alternative by using large language models to provide structured judgments on code quality. The goal is to investigate whether LLM-driven evaluation can approximate human assessment and support automated feedback for programming tasks.

## Project Goals

- Build a pipeline for evaluating code submissions with LLMs
- Compare pointwise and pairwise evaluation strategies
- Fine-tune language models on annotated code-quality datasets
- Measure agreement between model predictions and human labels
- Produce reproducible artifacts for thesis experiments

## Methodology

The repository implements a practical pipeline consisting of the following stages:

1. Data preparation
   - Load problem descriptions, code submissions, and human annotations
   - Construct evaluation datasets in pointwise and pairwise formats

2. Prompt-based assessment
   - Use structured prompts to elicit scores for correctness, efficiency, and readability
   - Generate model predictions through inference scripts

3. Fine-tuning
   - Fine-tune transformer-based language models for scoring tasks
   - Support parameter-efficient adaptation using LoRA and related techniques

4. Evaluation
   - Compare predicted scores against ground-truth or human labels
   - Report metrics such as quadratic weighted kappa, MAE, MSE, and RMSE

## Repository Structure

```text
code-score-eval/
├── data/
│   ├── calibration_set/
│   │   └── ... annotated and human-evaluated subsets
│   ├── finetune/
│   │   └── ... training labels for pointwise/pairwise tasks
│   ├── ground_truth/
│   │   └ ... gold labels for correctness, efficiency, and readability
│   ├── human_annotated_test_set/
│   │   └ ... human-labeled evaluation data
│   └── ... raw and processed dataset files
├── evaluation/
│   ├── inference/
│   │   └── ... inference and dataset utilities
│   ├── metric_pointwise_score.py
│   ├── metric_pairwise_score.py
│   ├── metric_pairwise_ranking.py
│   └── utils.py
├── finetune/
│   ├── finetune_pointwise_transformers.py
│   ├── finetune_pairwise_transformers.py
│   ├── prepare_pointwise_labels.py
│   ├── prepare_pairwise_labels.py
│   └── ... scripts for label balancing/undersampling
├── notebook/
│   └── ... notebooks for annotation, analysis, and experiment exploration
├── output/
│   ├── pointwise/
│   ├── pairwise/
│   └── human_evaluation/
├── fill_missing_choices.py
├── fill_missing_scores.py
├── gemini-inference.py
├── gemini-tokens-count.py
├── prompts.py
├── requirements.txt
├── split_original_train_test.py
└── README.md
```

## Installation

This project uses Python 3.10+ and depends on the packages listed in requirements.txt.

```bash
pip install -r requirements.txt
```

If you plan to run the training scripts on GPU, ensure that your CUDA-enabled environment is properly configured.

## Quick Start

### 1. Prepare data

The repository expects datasets in JSONL format under the data/ directory. The scripts in finetune/ and evaluation/ assume specific file structures and naming conventions.

### 2. Fine-tune a model

Example:

```bash
python finetune/finetune_pointwise_transformers.py --help
```

### 3. Run inference and evaluation

Evaluation scripts are available under the evaluation/ directory. You can inspect their CLI options and run them with the relevant input files.

## Example Workflow

```bash
# Example: inspect available training options
python finetune/finetune_pointwise_transformers.py --help

# Example: evaluate predictions against labels
python evaluation/metric_pointwise_score.py -i output/pointwise
```

## Notes for Thesis Use

This repository is intended as a research codebase rather than a polished production application. It is suitable for:

- reproducing experiments
- extending prompt templates
- adding new datasets
- comparing different model backbones
- documenting ablation studies

## Acknowledgment

This project was developed as part of a graduation thesis focused on leveraging large language models for automated code quality assessment.
