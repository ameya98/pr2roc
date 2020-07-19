# roc2pr

Resample and interconvert curves between ROC and PR space!  
The only requirement is to know the proportion of actual positives to actual negatives in the dataset.

## Terminology

* **TP**: True Positives: Actual positives labelled correctly.
* **TN**: True Negatives: Actual negatives labelled correctly.
* **FP**: False Positives: Actual negatives labelled incorrectly as positive.
* **FN**: False Negatives: Actual positives labelled incorrectly as negative.
* **TPR**: True Positive Rate = TP/(TP + FN).
* **FPR**: False Positive Rate = FP/(FP + TN).
* **Precision**: TP/(TP + FP).
* **Recall**: True Positive Rate = TP/(TP + FN).
* **ROC**: Reciever Operating Characteristic: A curve comparing the TPR (y-axis) as a function of the FPR (x-axis) for a given classifier at various thresholds.
* **PR**: Precision-Recall: A curve comparing the precision (y-axis) as a function of the recall (x-axis) for a given classifier at various thresholds.

## How does the resampling work?

There is a fundamental duality between ROC space and PR space. For a fixed dataset (specifically, a fixed ratio of actual positives to actual negatives), we can convert a ROC curve to a PR curve, and vice versa.

In ROC space, linear interpolation between points on a 'discrete' ROC curve is valid, because all points on the convex hull of a ROC curve are attainable. However, linear interpolation is not justified in PR space.

The solution is to:
* Convert a PR curve to its corresponding ROC curve, using the duality between the PR and ROC spaces.
* Interpolate the ROC curve. Here, we interpolate at equally spaced values of FPR, linearly interpolating the TPR values between adjacent points.
* Convert the interpolated ROC curve back to PR space, giving us an interpolated PR curve.

See the reference below for a more detailed description of the relationship between ROC curves and PR curves.  
This package is essentially an implementation of the methods in the paper.

## Usage

A snippet from [demo.ipynb](demo.ipynb):

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

## References

Jesse Davis and Mark Goadrich. 2006. *The relationship between Precision-Recall and ROC curves*.  
In Proceedings of the 23rd International Conference on Machine Learning (ICML ’06). Association for Computing Machinery, New York, NY, USA, 233–240.  
DOI: https://doi.org/10.1145/1143844.1143874
