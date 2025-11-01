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

import json
import torch
import random
from pathlib import Path
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

def get_results_dir(
    run_type: str
) -> Path:
    """
    E.g. 'sweep_6_JUL_2008_21:15' OR 'trial_8_JUN_2025_18:30'
    
    Args: 
        run_type: 'sweep' OR 'trial'

    Returns:
        path: path to results directory.
    """
    return Path(
        f"{run_type}_" 
        + datetime
            .now()
            .strftime(C.DATETIME_FMT)
            .upper()
    )

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
            results_json, 
            train_results, 
            eval_results, 
            res_id=0
        )
    
    elif not train_results: # test results
        log_epoch_results(
            results_json, 
            None, 
            eval_results, 
            res_id=0
        )

    else:
        for e_id in range(1, C.NUM_EPOCHS):
            log_epoch_results(
                results_json, 
                train_results, 
                eval_results, 
                res_id=e_id
            )


def log_epoch_results(
    results_json: Path,
    train_results: list[TrainResult] | None, 
    eval_results: list[EvalResult],
    res_id: int,
) -> None:
    # Get current file state
    with open(results_json, 'r') as f_in:
        data = json.load(f_in)

    # Construct file update
    root = data[results_json.stem][f'epoch_{res_id}']

    if train_results:
        root['train']['duration']      = train_results[res_id].duration
        root['train']['mean_accuracy'] = train_results[res_id].mean_accuracy
        root['train']['mean_loss']     = train_results[res_id].mean_loss

    label: str = 'val' if train_results else 'test'

    root[label]['duration']    = eval_results[res_id].duration
    root[label]['accuracy']    = eval_results[res_id].accuracy
    root[label]['loss']        = eval_results[res_id].loss
    root[label]['n_correct']   = eval_results[res_id].n_correct
    root[label]['tp']          = eval_results[res_id].tp
    root[label]['fn']          = eval_results[res_id].fn
    root[label]['tn']          = eval_results[res_id].tn
    root[label]['fp']          = eval_results[res_id].fp
    root[label]['precision']   = eval_results[res_id].precision
    root[label]['recall']      = eval_results[res_id].recall
    root[label]['f1_score']    = eval_results[res_id].f1_score
    root[label]['specificity'] = eval_results[res_id].specificity

    # Write file update
    with open(results_json, 'w') as f_out:
        json.dump(
            data,
            f_out,
            indent=4
        )
