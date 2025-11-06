# The MIT License (MIT)
# Copyright © 2025 Jake Weatherhead

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""General utility functions used throughout the pipeline."""

import json
import torch
import random
from pathlib import Path
from config import config
from datetime import datetime
import constants.constant as C
from schema.result import TrainResult, EvalResult


def format_time_delta(secs: float) -> str:
    if secs <= 0:
        return "0.00sec"
    
    mins: int = int((secs % C.SECONDS_IN_HOUR) // C.SECONDS_IN_MINUTE)
    secs: float = secs % C.SECONDS_IN_MINUTE

    if mins > 0:
        return f"{mins:02d}min {secs:05.2f}sec"
    return f"{secs:05.2f}sec"

def set_random_seeds() -> None:
    """Apply global seed to all components"""
    random.seed(C.GLOBAL_SEED)
    torch.manual_seed(C.GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(C.GLOBAL_SEED) 

def safe_divide(n: int, d: int) -> float:
    """
    Safely calculate n / d.

    Args:
        n: numerator.
        d: denominator.

    Returns:
        float: n / d, or 0.0 if d equals zero to avoid ZeroDivisionError.
    """
    return n / d if d else 0.0

def log_results(
    results_json: Path,
    train_results: list[TrainResult],
    eval_results: list[EvalResult]
) -> None:
    if train_results and len(train_results) == 1:
        log_epoch_results(
            results_json=results_json, 
            train_results=train_results, 
            eval_results=eval_results, 
            epoch_id=0
        )
    
    elif not train_results: # => test results
        log_epoch_results(
            results_json=results_json, 
            train_results=None, 
            eval_results=eval_results, 
            epoch_id=C.NUM_EPOCHS+1
        )

    else:
        for e_id in range(1, C.NUM_EPOCHS):
            log_epoch_results(
                results_json=results_json, 
                train_results=train_results, 
                eval_results=eval_results, 
                epoch_id=e_id
            )

from pathlib import Path
import json

def log_epoch_results(
    results_json: Path,
    train_results: list | None, 
    eval_results: list,
    epoch_id: int,
) -> None:
    with open(results_json, 'r') as f_in:
        data = json.load(f_in)

    root: dict  = data['results']
    label: str  = f'epoch_{epoch_id}' if train_results else 'test'
    epoch: dict = root.setdefault(label, {})

    if train_results:
        train             = epoch.setdefault('train', {})
        train['duration'] = train_results[0].duration
        train['accuracy'] = train_results[0].accuracy
        train['loss']     = train_results[0].loss

    eval_type = 'val' if train_results else 'test'
    eval                = epoch.setdefault(eval_type, {})
    eval['duration']    = eval_results[0].duration
    eval['accuracy']    = eval_results[0].accuracy
    eval['loss']        = eval_results[0].loss
    eval['n_correct']   = eval_results[0].n_correct
    eval['tp']          = eval_results[0].tp
    eval['fn']          = eval_results[0].fn
    eval['tn']          = eval_results[0].tn
    eval['fp']          = eval_results[0].fp
    eval['precision']   = eval_results[0].precision
    eval['recall']      = eval_results[0].recall
    eval['f1_score']    = eval_results[0].f1_score
    eval['specificity'] = eval_results[0].specificity

    with open(results_json, 'w') as f_out:
        json.dump(data, f_out, indent=4)
