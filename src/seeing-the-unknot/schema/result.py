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

from dataclasses import dataclass


@dataclass
class TrainResult:
    """Stores training result for a single epoch"""
    duration:      float
    mean_accuracy: float
    mean_loss:     float

@dataclass
class EvalResult:
    """Stores validation results for a single epoch OR test results.
    
    Confusion matrix bins:
        True Positives  (tp): Unknots classified as unknots.
        False Negatives (fn): Unknots classified as non-trivial knots.
        True Negatives  (tn): Non-trivial knots classified as non-trivial knots.
        False Positives (fp): Non-trivial knots classified as unknots.
    """
    duration:    float
    accuracy:    float
    loss:        float
    n_correct:     int 
    tp:            int             
    fn:            int             
    tn:            int             
    fp:            int            
    precision:   float
    recall:      float
    f1_score:    float
    specificity: float
