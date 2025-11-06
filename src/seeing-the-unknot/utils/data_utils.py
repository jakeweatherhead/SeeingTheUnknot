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

from typing import Any
import glob
import os
import random
from pathlib import Path
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision import transforms
import constants.constant as C
from config import config
from schema.dataset import KnotDataset

def get_transformation() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(C.IMAGENET_INPUT_DIMS),
        transforms.ToTensor(),
        transforms.Normalize(
            C.IMAGENET_CHANNEL_MEANS,
            C.IMAGENET_CHANNEL_STDS
        )
    ])

def _get_diagram_paths(
    split: str
) -> list[str]:
    m_path = f"../../knot_data/diagram/{split}/*.png"
    return glob.glob(m_path)

def _get_diagram_labels(
    paths: list[str]
) -> list[int]:
    return [
        C.KNOT_LABEL if Path(path).stem[0] == 'K' else C.UNKNOT_LABEL
        for path in paths
    ]

def get_labelled_diagram_paths(
    split: str
) -> tuple[list[str], list[int]]:
    paths  = _get_diagram_paths(split)
    labels = _get_diagram_labels(paths)
    paths, labels = _zip_shuffle(paths, labels)
    return paths, labels

def create_datasets(
    splits: dict[str, tuple[list[str], list[int]]]
) -> dict[str, KnotDataset]:
    datasets = {}

    for split, (image_paths, labels) in splits.items():
        datasets[split] = KnotDataset(
            image_paths=image_paths,
            labels=labels,
            transform=get_transformation()
        )

    return datasets

def create_dataloaders(
    config: config.Config,
    datasets: dict[str, KnotDataset]
) -> dict[str, DataLoader]:
    common = {
        "batch_size": config.batch_size,
        "pin_memory": True,
        "num_workers": os.cpu_count(),
    }

    dataloaders: dict[str, DataLoader] = {
        split: DataLoader(
            datasets[split],
            **{**common, "shuffle": (split == 'train')}
        ) for split in C.SPLITS
    }
    
    return dataloaders

def get_dataset_summary(
    splits: dict[str, tuple[list[str], list[int]]]
) -> dict[str, Any]:
    summary: dict = {}
    
    for split, (paths, labels) in splits.items():
        label_count  = lambda label: sum(1 for l in labels if l == label)
        knot_count   = label_count(C.KNOT_LABEL)
        unknot_count = label_count(C.UNKNOT_LABEL)
        
        summary[split] = {
            'total':             len(paths),
            'knots':             knot_count,
            'unknots':           unknot_count,
            'knot_percentage':   (knot_count / len(paths)) * 100,
            'unknot_percentage': (unknot_count / len(paths)) * 100
        }
    
    return summary

def _plot_curve(
    model: str,
    train_data: list[float], 
    val_data: list[float],
    config: dict,
    results_dir: Path
) -> None:
    epochs: range = range(0, len(train_data))
    plt.plot(epochs, 
             train_data, 
             label=config['train_label'], 
             color='blue', 
             linestyle='--') 
    
    plt.plot(epochs, 
             val_data, 
             label=config['val_label'], 
             color='green')
    
    plt.xlabel(config['x_label'])
    plt.ylabel(config['y_label'])
    plt.title(config['title'](model.upper()))
    plt.legend()
    plt.savefig(results_dir / config['filename'],
                dpi=C.DPI)
    plt.clf()

def _zip_shuffle(
    lst1: list[str], 
    lst2: list[int]
) -> tuple[list[str], list[int]]:
    combined: list = list(zip(lst1, lst2))
    random.shuffle(combined)
    lst1, lst2 = zip(*combined)
    
    return lst1, lst2
