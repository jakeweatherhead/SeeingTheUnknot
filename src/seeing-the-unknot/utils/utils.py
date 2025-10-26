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

import torch
import random
from pathlib import Path
from datetime import datetime
import constants.constant as C


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
    """E.g. 'sweep_6_JUL_2008_21:15' OR 'trial_8_JUN_2025_18:30'
    
    Args: 
        run_type: 'sweep' OR 'trial'

    Returns:
        Path: path to results directory.
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
        float: n / d, or 0.0 if d equals zero to avoid ZeroDivisionError
    """
    return n / d if d else 0.0