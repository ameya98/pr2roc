# roc2pr

Resample and interconvert curves between ROC and PR space!  
The only requirement is to know the proportion of actual positives to actual negatives in the dataset.

## Terminology

* **ROC**: Reciever Operating Characteristic.
* **PR**: Precision-Recall.
* **TP**: True Positives: Actual positives labelled correctly.
* **TN**: True Negatives: Actual negatives labelled correctly.
* **FP**: False Positives: Actual negatives labelled incorrectly as positive.
* **FN**: False Negatives: Actual positives labelled incorrectly as negative.
* **TPR**: True Positive Rate = TP/(TP + FN).
* **FPR**: False Positive Rate = FP/(FP + TN).
* **Precision**: TP/(TP + FP).
* **Recall**: True Positive Rate = TP/(TP + FN).

## How does the resampling work?

In ROC space, linear interpolation is valid. However, this is not true in PR space.
The solution is to:
* Convert a PR curve to its corresponding ROC curve, using the duality between the PR and ROC spaces.
* Interpolate the ROC curve. Here, we interpolate at equally spaced values of FPR, linearly interpolating the TPR values between adjacent points.
* Convert the interpolated ROC curve back to PR space, giving us an interpolated PR curve.

## Usage

A snippet from *demo.ipynb*:

```python
from roc2pr import PRCurve
from numpy import allclose

# Define points on precision-recall curve.
recall_vals = [0.25, 0.4, 0.5]
precision_vals = [0.5, 0.3, 0.25]
pr_points = zip(recall_vals, precision_vals)

# Create the curve.
pr = PRCurve(pr_points, pos_neg_ratio=0.25)

# Resample, to get another precision-recall curve, with 100 points!.
pr_sampled = pr.resample(num_points=100)

# Convert to a ROC curve.
roc = pr.to_roc()

# We can resample this ROC curve, as well!
roc_sampled = roc.resample(num_points=100)

# Get the points (as a list of 2-tuples) on any curve with the *.points()* method.
pr_points = pr.points()
pr_sampled_points = pr_sampled.points()
roc_points = roc.points()
roc_sampled_points = roc_sampled.points()

# You can re-convert the ROC curve back to a PR curve!
pr_reconverted = roc.to_pr()

# This should pass.
assert allclose(pr_reconverted.points(), pr.points())
```