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
import pyhopper
import os

from schema.result import EvalResult
from config import config
from typing import List

import constants.constant as C

class Sweeper:
    """Hyperparameter optimization sweeps comprise two or more Trials"""
    history: List[EvalResult]
    best_val_loss: float
    best_initial_params: config.Config
    
    def __init__(self):
        self.architecture: str = os.getenv('ARCHITECTURE')
        self.config: config.Config = config.OPTIONS[self.architecture]
        self.device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def sweep(self) -> None:
        from model.trial import Trial

        sweep = pyhopper.Search(C.SWEEP_SEARCH_SPACE)
        trial: Trial = Trial(run_type='sweep')

        try:
            sweep.run(
                objective_function=trial.run,
                direction=C.SWEEP_DIRECTION,
                runtime=C.SWEEP_DURATION,
                n_jobs=C.SWEEP_N_JOBS
            )

        except Exception as e:
            raise Exception(f"Sweeper.sweep() exception: {e}")

    @classmethod
    def update_history(
        cls, 
        final_val_result: EvalResult,
        initial_params
    ) -> None:
        """Appends latest results to sweep history. Updates best loss if new best reached"""
        cls.history.append(final_val_result)
        if final_val_result.loss < cls.best_val_loss:
            cls.best_val_loss = final_val_result.loss
            cls.best_initial_params = initial_params.copy()