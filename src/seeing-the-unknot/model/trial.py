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

from schema.dataset import KnotDataset
from typing import Dict, Any
from pathlib import Path
from dataclasses import replace

import os
import json
import torch
from torch import optim
from torch import device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module

from schema.result import TrainResult, EvalResult
from model.train import train
from model.evaluate import evaluate

from config import config
import constants.constant as C
from datetime import datetime, timezone
from utils import utils, cam_utils, data_utils, model_utils

class Trial:
    def __init__(self) -> None:
        self.run_type: str                    = "trial"
        self.architecture: str                = os.environ['ARCHITECTURE']
        self.config: config.Config            = config.OPTIONS[self.architecture]
        self.trial_dir: Path                  = self._get_trial_dir()
        self.device: device                   = self._get_device()
        self.criterion: Module                = self._get_criterion()
        self.splits: dict                     = self._get_splits()
        self.model: Module                    = model_utils.build_model(config, self.device)
        self.datasets: Dict[str, KnotDataset] = data_utils.create_datasets(self.splits)
        self.dataloaders: dict                = data_utils.create_dataloaders(self.config, self.datasets)      

    def run(self) -> float:
        """
        Runs a single Trial collecting results from epoch 0 through the final epoch.
        
        Args:
            params: updated hyperparameters passed by PyHopper between Trials

        Returns:
            float: performance metric to be optimized by PyHopper.Search() in the 
                   direction specified in constants/constant.py (SWEEP_DIRECTION)
        """
        try:
            config: dict            = replace(self.config)
            optimizer: Optimizer    = self._get_optimizer(self.model, config)
            scheduler: _LRScheduler = self._get_scheduler(optimizer, config)

            train_result: TrainResult = evaluate(
                model=self.model,
                dataloader=self.dataloaders['train'],
                device=self.device,
                criterion=self.criterion
            )

            val_result: EvalResult = evaluate(
                model=self.model,
                dataloader=self.dataloaders['val'],
                device=self.device,
                criterion=self.criterion
            )

            utils.log_results( # Log epoch zero results
                results_json=self.json_f, 
                train_results=[train_result], 
                eval_results=[val_result]
            )

            train_results, val_results = train(
                device=self.device,
                model=self.model, 
                dataloaders=self.dataloaders, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                config=config,
                criterion=self.criterion,
                results_dir=self.trial_dir
            )

            utils.log_results(
                results_json=self.json_f, 
                train_results=train_results, 
                eval_results=val_results
            )

            if self.run_type == 'trial':
                data_utils.save_curve_plots(
                    config=config, 
                    trial_results=self.trial_results, 
                    results_dir=self.trial_dir
                )

            self._run_tests()

        except Exception as e:
            raise Exception(f"Error in Trial: {e}") 

    def _get_trial_dir(self) -> Path:
        """
        Creates a directory to store the current Trial's results.
        
        Returns:
            Path: the directory to store results in for the current Trial.
        """
        trial_dir: Path = Path(
            self.config.get_run_path(),
            utils.get_results_dir(
                run_type=self.run_type
            )
        )
        os.makedirs(trial_dir, exist_ok=True)

        return trial_dir
    
    def _get_splits(self) -> dict[str, tuple[list[str], list[int]]]:
        """
        Creates a mapping of data splits to labelled image paths for all splits
        specified in config/config.py.
        
        Returns:
            tuple[list[str], list[int]]: mapping of data splits to labelled images.    
        """
        return {
            split: data_utils.get_labelled_diagram_paths(split) 
            for split in C.SPLITS
        }

    def _get_optimizer(
        self,
        model: Module, 
        config: config.Config
    ) -> Optimizer:
        """
        Selects the optimizer specified in config/config.py.
        
        Args:
            model: model to be fine-tuned.
            config: user-specified configurations.
            
        Returns:
            torch.optim.Optimizer: optimizer to be used to fine-tune the model during Trial.
        """
        config = config or {}
        common: dict = {
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay
        }
        
        optimizers: dict = {
            "adam": {
                "class": optim.Adam,
                "params": {
                    **common,
                    "eps": config.eps,
                    "betas": (
                        config.beta1, 
                        config.beta2
                    )
                }
            },
            "adamw": {
                "class": optim.AdamW,
                "params": {
                    **common,
                    "eps": config.eps,
                    "betas": (
                        config.beta1, 
                        config.beta2
                    )
                }
            }
        }
        
        optimizer_config: dict = optimizers[config.optimizer]
        
        optimizer: Optimizer = optimizer_config["class"](
            model.parameters(), 
            **optimizer_config["params"]
        )

        return optimizer
        
    def _get_scheduler(
        self,
        optimizer: Optimizer, 
        config: config.Config,
    ) -> _LRScheduler:
        """
        Selects the learning-rate scheduler specified in config/config.py.
        
        Args:
            model: model to be fine-tuned.
            config: user-specified configurations.
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler: learning-rate scheduler to be 
            used to fine-tune the model during Trial.
        """
        config = config or {}
        
        schedulers: dict[str, dict[str, Any]] = {
            "step": {
                "class": optim.lr_scheduler.StepLR,
                "params": {
                    "step_size": config.step_size,
                    "gamma": config.gamma
                }
            },
            "cosine": {
                "class": optim.lr_scheduler.CosineAnnealingLR,
                "params": {
                    "T_max": config.num_epochs,
                    "eta_min": config.cosine_eta_min
                }
            },
            "exponential": {
                "class": optim.lr_scheduler.ExponentialLR,
                "params": {
                    "gamma": config.gamma
                }
            }
        }
            
        scheduler_config: dict[str, Any] = schedulers[config.scheduler]
        
        scheduler: _LRScheduler = scheduler_config["class"](
            optimizer,
            **scheduler_config["params"]
        )

        return scheduler

    def _run_tests(self) -> None:
        ckpt_paths: list[Path] = model_utils.ckpt_paths(self.trial_dir)

        grad_cam_params: dict = {
            "dataset": self.datasets['test'],
            "ckpt_paths": ckpt_paths
        }

        cam_utils.generate_saliency_maps(
            self.config, 
            **grad_cam_params
        )

        for ckpt_path in ckpt_paths:
            model: Module = model_utils.load_model(self.model, ckpt_path)
            test_result: EvalResult = evaluate(
                model=model,
                dataloader=self.dataloaders['test'],
                device=self.device,
                criterion=self.criterion
            )
            
            utils.log_results(
                results_json=self.json_f, 
                train_results=None, 
                eval_results=test_result
            )


    def _get_device(self) -> device:
        """
        Initializes the device on which the model will be fine-tuned.
        Prefers CUDA GPU but defaults to CPU if a CUDA GPU is not available.
        """
        return device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def _get_criterion(self) -> Module:
        return torch.nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
    
    def _get_trial_id(self) -> str:
        """Creates unique trial identifier in trial_DD/MMM/YYYY-HH:MM:SS format."""
        dt = datetime.now(timezone.utc).strftime("%d/%b/%Y-%H:%M:%S").upper()
        return f"trial_{dt}"
    
    def _create_results_assets(self) -> None:
        # Make results directory for this Trial
        os.mkdir(
            (results_dir := self.config.get_git_root() / self.trial_id), 
            exist_ok=True
        )

        # Create JSON results file
        self.json_f: str = f"{results_dir}/{self.trial_id}_results.json"

        with open(f"{results_dir}/{self.trial_id}_results.json", 'w') as f:
            data = json.load(f)

        data: dict = {f"{self.trial_dir}_results": {}}

        with open(self.json_f, 'w') as f_out:
            json.dump(data, f_out, indent=4)

        # Create saliency map directories
        os.mkdir(smap_dir := results_dir / 'smaps', exist_ok=True)
        os.mkdir(smap_dir / 'TP', exist_ok=True)
        os.mkdir(smap_dir / 'FN', exist_ok=True)
        os.mkdir(smap_dir / 'TN', exist_ok=True)
        os.mkdir(smap_dir / 'FP', exist_ok=True)
