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
from torch.nn import Module
from torch.utils.data import DataLoader

import time
from schema.result import EvalResult
import constants.constant as C
from utils import utils


def evaluate(
    model: Module,
    dataloader: DataLoader,
    device: device,
    criterion: Module,
) -> EvalResult:
    """
    Evaluate knot classifier on 'validation' OR 'test' split.

    We studied the unknot decision problem where the positive instance 
    was the unknot. Non-trivials were treated as the negative instance.

    Confusion matrix bins:
        True Positives  (tp): Unknots classified as unknots.
        False Negatives (fn): Unknots classified as non-trivial knots.
        True Negatives  (tn): Non-trivial knots classified as non-trivial knots.
        False Positives (fp): Non-trivial knots classified as unknots.

    Args:
        model:      Model to evaluate.
        dataloader: Stores images and paths to those images.
        device:     Device to evaluate models on.
        criterion:  Loss function applied to outputs and labels.

    Returns:
        EvalResult: Dataclass containing evaluation results
    """
    ts = time.time()

    model.eval()
    
    total_loss = 0.0
    
    tp = 0
    fn = 0
    tn = 0
    fp = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predictions = torch.max(outputs.data, (ROW_DIMENSION := 1))
            
            total_loss += (loss := loss.item())
            
            for i in range(labels.size(0)):
                true_label = labels[i].item()
                pred_label = predictions[i].item()
                
                K, U = C.KNOT_LABEL, C.UNKNOT_LABEL
                tp += (true_label, pred_label) == (U, U)
                fn += (true_label, pred_label) == (U, K)
                tn += (true_label, pred_label) == (K, K)
                fp += (true_label, pred_label) == (K, U)
    
    accuracy = ((tp + tn) / len(dataloader.dataset)) * 100

    precision   = utils.safe_divide(tp, tp + fp)
    recall      = utils.safe_divide(tp, tp + fn)
    specificity = utils.safe_divide(tn, tn + fp)
    f1_score    = utils.safe_divide(2 * tp, 2 * tp + fp + fn)

    return EvalResult(
        duration=time.time() - ts,
        accuracy=accuracy,
        loss=loss,
        correct_count=(tp+tn),
        tn=tn,
        fp=fp,
        tp=tp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        specificity=specificity
    )
    