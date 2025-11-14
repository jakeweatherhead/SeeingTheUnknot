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
import torch.distributed as dist
from torch import optim
from torch import device
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

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
        self.trial_id: str                    = self._get_trial_id()
        self.trial_dir: Path                  = self._create_trial_dir()
        self.device: device                   = self._get_device()
        self.criterion: Module                = self._get_criterion()
        self.splits: dict                     = self._get_splits()
        self.model: Module                    = model_utils.build_model(self.config, self.device)
        if dist.is_available() and dist.is_initialized():
            self.model = FSDP(self.model)
        self.datasets: Dict[str, KnotDataset] = data_utils.create_datasets(self.splits)
        self.dataloaders: dict                = data_utils.create_dataloaders(self.config, self.datasets)
        self.rank: int                        = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if self.rank == 0:
            self._create_results_json()

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
                criterion=self.criterion,
                epoch_zero=True
            )

            val_result: EvalResult = evaluate(
                model=self.model,
                dataloader=self.dataloaders['val'],
                device=self.device,
                criterion=self.criterion
            )

            if self.rank == 0:
                with open(f'{self.trial_dir}/log.txt', "a") as f:
                    f.write(f"Epoch 0: Train Acc: {train_result.accuracy}, Val Acc: {val_result.accuracy}.\n")

            if self.rank == 0:
                utils.log_results(
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
                criterion=self.criterion,
                trial_dir=self.trial_dir
            )

            if self.rank == 0:
                utils.log_results(
                    results_json=self.json_f, 
                    train_results=train_results, 
                    eval_results=val_results
                )

            if self.rank == 0:
                self._run_tests()

        except Exception as e:
            raise Exception(f"Error in Trial: {e}") 

    def _create_trial_dir(self) -> Path:
        """
        Creates a directory to store the current Trial's results.
        
        Returns:
            Path: the directory to store results in for the current Trial.
        """
        trial_dir: Path = self.config.get_run_path() / self.trial_id
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
                    "T_max": C.NUM_EPOCHS,
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

        # cam_utils.generate_saliency_maps( # NOTE: smaps will be created locally after run completion
        #     self.config, 
        #     **grad_cam_params
        # )

        for ckpt_path in ckpt_paths:
            model: Module = model_utils.load_model(self.model, Path(ckpt_path))
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
        use_cuda = torch.cuda.is_available()
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if use_cuda and world_size > 1 and dist.is_available() and not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            torch.cuda.set_device(local_rank)
            return device(f"cuda:{local_rank}")
        if use_cuda:
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            if local_rank < torch.cuda.device_count():
                torch.cuda.set_device(local_rank)
                return device(f"cuda:{local_rank}")
            return device("cuda")
        return device("cpu")
    
    def _get_criterion(self) -> Module:
        return torch.nn.CrossEntropyLoss(
            label_smoothing=self.config.label_smoothing
        )
    
    def _get_trial_id(self) -> str:
        """
        Creates a unique trial identifier in the form of: trial_DD_MMM_YYYY-HH:MM:SS.
        """
        dt = datetime.now().strftime(C.DATETIME_FMT).upper()
        return f"trial_{dt}"

    def _create_results_json(self) -> None:
        self.json_f: str = self.trial_dir / f"{self.trial_id}_results.json"

        data: dict = {f"results": {}}

        with open(self.json_f, 'w') as f_out:
            json.dump(data, f_out, indent=4)

    def _mk_results_dir(self) -> None:
        results_dir = self.config.get_run_path() / self.trial_id
        os.mkdir(results_dir)

        return results_dir

    def _create_smaps_dir(
        self,
        results_dir
    ) -> None:
        os.mkdir(smap_dir := results_dir / 'smaps', exist_ok=True)
        os.mkdir(smap_dir / 'TP', exist_ok=True)
        os.mkdir(smap_dir / 'FN', exist_ok=True)
        os.mkdir(smap_dir / 'TN', exist_ok=True)
        os.mkdir(smap_dir / 'FP', exist_ok=True)