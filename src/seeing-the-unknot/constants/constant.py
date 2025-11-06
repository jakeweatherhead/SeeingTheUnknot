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

import pyhopper

DPI: int = 300
NUM_EPOCHS: int = 30
NUM_CLASSES: int = 2
KNOT_LABEL: int = 0    # negative instance
UNKNOT_LABEL: int = 1  # positive instance
GLOBAL_SEED: int = 16011997
PATIENCE: int = 5
SPLITS: list[str] = ['train', 'val', 'test']
SECONDS_IN_HOUR: int = 3_600
SECONDS_IN_MINUTE: int = 60
DATETIME_FMT: str = '%d_%b_%Y_%H:%M:%S'
LOSS_PLOT_FILENAME: str = 'loss_plot.png'
ACC_PLOT_FILENAME: str = 'acc_plot.png'

IMAGENET_INPUT_DIMS: tuple[int, int] = (224, 224)
IMAGENET_CHANNEL_MEANS: list[float] = [0.485, 0.456, 0.406]
IMAGENET_CHANNEL_STDS: list[float] = [0.229, 0.224, 0.225]

SWEEP_N_JOBS: int = 1
SWEEP_DURATION: str = '24h'
SWEEP_DIRECTION: str = 'minimize'
SWEEP_SEARCH_SPACE: dict = {
    'learning_rate': pyhopper.float(1e-5, 1e-1, '0.1g'),
    'batch_size': pyhopper.choice([8, 16, 32]),
    'num_epochs': pyhopper.int(50, 75),
    'weight_decay': pyhopper.float(1e-6, 1e-2, '0.1g'),
    'eps': pyhopper.float(1e-9, 1e-6, '0.1g'),
    'beta_1': pyhopper.float(0.85, 0.95, '0.3f'),
    'beta_2': pyhopper.float(0.95, 0.999, '0.3f'),
    'scheduler': pyhopper.choice(['step', 'cosine', 'exponential']),
    'step_size': pyhopper.int(10, 50),
    'gamma': pyhopper.float(0.1, 0.8, '0.2f'),
    'cosine_eta_min': pyhopper.float(1e-7, 1e-4, '0.1g'),
    'label_smoothing': pyhopper.float(0.0, 0.2, '0.2f'),
}

LOSS_PLOT_CONFIG: dict = {
    'train_label': 'Train Loss',
    'train_linestyle': '--',
    'val_label': 'Val Loss',
    'x_label': 'Number of Epochs',
    'y_label': 'Error (Loss)',
    'title': lambda model: f"{model} Train/Val Loss",
    'filename': 'loss_plot.png'
}

ACC_PLOT_CONFIG: dict = {
    'train_label': 'Train Accuracy',
    'train_linestyle': '--',
    'val_label': 'Val Accuracy',
    'x_label': 'Number of Epochs',
    'y_label': 'Accuracy',
    'title': lambda model: f"{model} Train/Val Accuracy",
    'filename': 'accuracy_plot.png'
}
