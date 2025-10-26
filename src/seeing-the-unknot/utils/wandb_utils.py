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

# This module is not used in the current state of the codebase but may be useful for future iterations.

from schema.result import TrainResult, EvalResult
from dataclasses import asdict
import constants.constant as C
from torch.nn import Module
from config import config
from pathlib import Path
import wandb

def start(
    config: config.Config,
    trial_dir: Path
) -> None:
    wandb.login()
    wandb.init(
        project=config.wandb_project_name,
        entity=config.wandb_entity,
        name=trial_dir.name, # final path component
        config=asdict(config),
    )

def log_epoch_zero(
    train_result: TrainResult, 
    val_result: EvalResult
) -> None:
    wandb.log({
        "epoch": 0,
        "train/loss":       train_result.final_loss,
        "train/acc":        train_result.final_accuracy,
        "val/loss":         val_result.loss,
        "val/acc":          val_result.accuracy,
        "val/precision":    val_result.precision,
        "val/recall":       val_result.recall,
        "val/f1":           val_result.f1_score,
        "val/specificity":  val_result.specificity,
    }, step=0)

def log_test_results(
    test_result: EvalResult
) -> None:
    wandb.log({
        "test/acc":         test_result.accuracy,
        "test/loss":        test_result.loss,
        "test/precision":   test_result.precision,
        "test/recall":      test_result.recall,
        "test/f1":          test_result.f1_score,
        "test/specificity": test_result.specificity,
        "test/tp":          test_result.tp,
        "test/tn":          test_result.tn,
        "test/fp":          test_result.fp,
        "test/fn":          test_result.fn,
    }) 

def watch(
    model: Module
) -> None:
    wandb.watch(
        model, 
        log='all', 
        log_freq=10
    )

def log_curve_plots(
    trial_dir: Path
) -> None:
    wandb.log({
        "plots/loss": wandb.Image(trial_dir / C.LOSS_PLOT_FILENAME),
        "plots/accuracy": wandb.Image(trial_dir / C.ACC_PLOT_FILENAME),
    })
