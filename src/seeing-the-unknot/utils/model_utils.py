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
from torch import device
from torch.nn import Module, Linear

from datetime import datetime
from pathlib import Path
from typing import Dict
from config import config

import timm


def build_model(
    config: config.Config,
    device: device
) -> Module:
    """
    Download pretrained weights from Timm libary if not locally present
    else, load pretrained weights from a local directory.
    
    Args:
        config: global configuration set
        device: 'cpu' or 'gpu', device to use for PyTorch operations
    """
    try:
        local_weights_path: Path = _get_local_weights_path(config)
        use_local: bool = local_weights_path.exists() 
        
        model = timm.create_model(model_name=config.model_name, 
                                  pretrained=(not use_local), # download weights if not locally present
                                  num_classes=config.num_classes)

        if use_local:
            _load_local_weights(model, local_weights_path)
        
        _adapt_classifier(model, config.num_classes)

        return model.to(device)
        
    except Exception as e:
        raise Exception(f"Failed to build model {config.model_name}: {e}")

def save_model(
    model: Module,
    results_dir: Path, 
    metrics: Dict[str, float],
) -> Path:
    """
    Save model and return path to the newly saved weights.
    
    Args:
        model      : PyTorch model instance
        results_dir: path to sweep/trial results directory
        metrics    : contains validation accuracy of model to save
    """
    try:
        pth_file_basename = f"val_acc{metrics['val_acc']:.3f}".replace('.', '_')
        model_dir = f"{datetime.now().strftime('%H_%M_%S')}_{pth_file_basename}"
        results_dir = results_dir / model_dir
        results_dir.mkdir(parents=True, exist_ok=True)
        filename = pth_file_basename + ".pth"
        weights_path = results_dir / filename
        torch.save(model.state_dict(), weights_path)

        return weights_path  # reference for testing phase
        
    except Exception as e:
        error_msg = f"Model Utils: save_model() Error saving model: {e}"
        raise Exception(error_msg)

def load_model(
    model: Module, 
    model_path: str
) -> Module:
    """
    Return model instance loaded with local pretrained weights
    
    Args:
        model     : PyTorch model instance
        model_path: path to pretrained weights
    """
    try:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        return model
        
    except Exception as e:
        raise Exception(f"Model Utils: load_model() error loading model: {e}")
    
def ckpt_paths(
    results_dir: Path
) -> list[Path]:
    """
    Returns a list of paths to all model checkpoint files saved in the current Trial.
    
    Args:
        results_dir: root directory of all model checkpoints saved in the current Trial.
    """
    return [f.resolve() for f in results_dir.glob("*/*.pth")]

def _get_local_weights_path(
    config: config.Config
) -> Path:
    """
    Returns a system Path to local, pretrained weights.
    
    Args:
        config: global configuration option
    """
    return Path(
        config.timm_model_path, 
        config.model_name, 
        config.timm_weights_filename
    )

def _classifier_head_mismatch(
    model: Module, 
    num_classes: int = 2
) -> bool:
    """
    Return True if model has a fully-connected layer, and the 
    fully-connected layer is an instance of Linear and the label 
    space cardinality does not equal two (non-trivial knots, unknots)
    else False.

    Args:
        model      : instance of a PyTorch model
        num_classes: label space cardinality
    """
    return (hasattr(model, 'fc')                       # has fully-connected layer
            and isinstance(model.fc, Linear)           # fc layer is Linear
            and model.fc.out_features != num_classes)  # wrong number of classes


def _classifier_reset_required(
    model: Module, 
    num_classes: int = 2
) -> bool:
    """
    Return False if model has a fully-connected layer,
    and the fully-connected layer is an instance of Linear
    and the label space cardinality equals two (non-trivial knots, unknots)
    else True.

    Args:
        model      : instance of a PyTorch model
        num_classes: label space cardinality
    """
    return not (hasattr(model, 'fc')                     
                and isinstance(model.fc, Linear) 
                and model.fc.out_features == num_classes)

def _load_local_weights(
    model: Module,
    path: Path,
) -> None:
    """
    Load and apply local, pretrained weights to model instance
    
    Args:
        model: PyTorch model instance
        path : path to local, pretrained weights
    """
    try:
        local_state_dict = torch.load(path, map_location='cpu')
        model_dict = model.state_dict()
        
        filtered_state_dict = {
            key: value 
            for key, value 
            in local_state_dict.items()
            if key in model_dict and model_dict[key].shape == value.shape
        }
        
        model.load_state_dict(filtered_state_dict, strict=False)

    except Exception as e:
        raise Exception(f"Failed to load local weights: {e}")
    
def _adapt_classifier(
    model: Module, 
    num_classes: int = 2
) -> None:
    """
    Replace the fully-connected layer with a new Linear instance
    matching the label space cardinality of two.
    
    Args:
        model      : PyTorch model instance
        num_classes: label space cardinality
    """
    try:
        if _classifier_head_mismatch(model, num_classes):
            num_features = model.fc.in_features
            model.fc = Linear(num_features, num_classes)
            
        elif _classifier_reset_required(model, num_classes):
            try:
                model.reset_classifier(num_classes)
            except Exception as e:
                raise RuntimeError(f"Could not adapt classifier: {e}")
            
    except Exception as e:
        raise RuntimeError(f"Could not adapt classifier: {e}")