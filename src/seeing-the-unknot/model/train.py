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
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import device
from torch.nn import Module
from torch.optim.lr_scheduler import _LRScheduler

from utils import model_utils
from config import config
from pathlib import Path
from typing import Dict
from utils import model_utils
from schema.result import TrainResult, EvalResult
from model.evaluate import evaluate

import constants.constant as C
import time


def train(
    model: Module,
    dataloaders: Dict[str, DataLoader],
    device: device,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    config: config.Config,    
    criterion: Module,
    results_dir: Path
) -> tuple[list[TrainResult], list[EvalResult]]:
    train_results: list[TrainResult] = []
    val_results: list[EvalResult]    = []
    losses: list[float]              = []
    accs: list[float]                = []
    patience: int                    = C.PATIENCE
    best_val_accuracy: float         = float('-inf')
    n_train: int                     = len(dataloaders['train'].dataset)

    model.train()

    for epoch_idx in range(config.num_epochs):
        ts                = time.time()
        total_loss: float = 0.0
        n_correct: int    = 0
        n_processed: int  = 0

        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(inputs)

            _, predictions = torch.max(outputs, (ROW_DIMENSION := 1))
            loss: float = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data).item()
            samples_processed += inputs.size(0)

        train_size: int = len(dataloaders['train'].dataset)
        losses.append(final_loss := running_loss / train_size)
        accs.append(final_acc := running_corrects / train_size)

        train_result: TrainResult = TrainResult(
            duration=time.time() - ts,
            final_accuracy=final_acc,
            final_loss=final_loss
        )

        train_results.append(train_result)

        val_result: EvalResult = evaluate(
            model=model,
            dataloader=dataloaders['val'],
            device=device,
            criterion=criterion
        )

        val_results.append(val_result)

        if val_result.accuracy > best_val_accuracy:
            best_val_accuracy = val_result.accuracy
            patience = C.PATIENCE

            weights_path = model_utils.save_model(
                model=model,
                results_dir=results_dir,
                metrics={'val_acc': best_val_accuracy},
            )
        else:
            patience -= 1

        if patience == 0:
            print(f"Patience exceeded; Stopping! (patience={C.PATIENCE} epochs)")

        scheduler.step()

        print(f"Training epoch {epoch_idx} complete...")

    return train_results, val_results
