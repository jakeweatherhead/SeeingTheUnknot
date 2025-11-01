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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import subprocess
import constants.constant as C

@dataclass(frozen=True, kw_only=True)
class Config(ABC):
    num_classes:           int = C.NUM_CLASSES
    seed:                  int = C.GLOBAL_SEED
    timm_model_path:       str = "<PRETRAINED-WEIGHTS-PATH>"
    timm_weights_filename: str = "pytorch_model.bin"

    @abstractmethod
    def get_run_path() -> Path: ...
    
    @staticmethod
    def get_git_root():
        """
        Runs git rev-parse to get path to the SeeingTheUnknot .git directory. 
        Assumes directory structure follows that of the remote repository
        at https://github.com/jakeweatherhead/SeeingTheUnknot.
        """
        out = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True, 
            capture_output=True, 
            text=True
        )
        
        return Path(out.stdout.strip()).resolve()


@dataclass(frozen=True, kw_only=True)
class CNNConfig(Config):
    learning_rate:       float = 9e-05
    batch_size:            int = 8
    weight_decay:        float = 2e-06
    eps:                 float = 1e-09
    optimizer:             str = "adam"
    beta1:               float = 0.928
    beta2:               float = 0.999
    scheduler:             str = "cosine"
    step_size:             int = 38
    gamma:               float = 0.8
    cosine_eta_min:      float = 5e-05
    label_smoothing:     float = 0.1
    patience:              int = 5
    architecture:          str = "cnn"
    model_name:            str = "timm/convnext_base.fb_in22k_ft_in1k"

    def get_run_path() -> Path:
        return Config.get_git_root() / "runs" / "cnn"

@dataclass(frozen=True, kw_only=True)
class ViTConfig(Config):
    learning_rate:       float = 3e-05
    batch_size:            int = 8
    weight_decay:        float = 0.05
    eps:                 float = 4e-09
    optimizer:             str = "adamw"
    beta1:               float = 0.9
    beta2:               float = 0.999
    scheduler:             str = "step"
    step_size:             int = 10
    gamma:               float = 0.57
    cosine_eta_min:      float = 2e-07
    label_smoothing:     float = 0.001
    patience:              int = 5
    architecture:          str = "vit"
    model_name:            str = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"

    def get_run_path() -> Path:
        return Config.get_git_root() / "runs" / "vit"


OPTIONS: dict[str, Config] = {
    "cnn": CNNConfig(),
    "vit": ViTConfig(),
}

