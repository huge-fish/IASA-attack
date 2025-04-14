# IASA - Implementation and Evaluation

## Overview

This repository contains the implementation of the IASA algorithm along with evaluation frameworks for both the proposed method and baseline approaches.

## Repository Structure

- `eval.py`: Main entry point for the evaluation framework
- `rs_attack_section/`: Contains the core implementation of the IASA algorithm
- `eval_base/`: Implements evaluation for baseline methods
- `assets/`: Contains visual resources, including the algorithm flow chart

## Algorithm Flow

The IASA algorithm implementation follows the flow chart as illustrated below:

![IASA Algorithm Flow](assets/流程.png)

## Getting Started
```python
    pip install -r requirements.txt
```
## Usage
To run the evaluation:
```python
python eval.py
```
For baseline evaluation:

```python
python eval_base.py
```

## PARAMETER
```python
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--norm', type=str, default='L0')
    parser.add_argument('--k', default=50., type=float)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--loss', type=str, default='margin')
    parser.add_argument('--model', default='pt_defense', type=str)
    parser.add_argument('--n_ex', type=int, default=320)
    parser.add_argument('--attack', type=str, default='rs_attack')
    parser.add_argument('--n_queries', type=int, default=500)
    parser.add_argument('--targeted', action='store_true')
    parser.add_argument('--target_class', type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--constant_schedule', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--use_feature_space', action='store_true')

  
```
