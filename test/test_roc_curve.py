from __future__ import division

from pr2roc import ROCCurve

import numpy as np
import pytest
from pytest import approx

def test_roc_curve_conversion():
    fpr_vals = [40/60, 45/60]
    tpr_vals = [100/120, 110/120]
    points = zip(fpr_vals, tpr_vals)
    roc_curve = ROCCurve(points, 120/60, label='Test ROC Curve')

    pr_curve = roc_curve.to_pr()
    rec_actual = [point[0] for point in pr_curve.points()]
    prec_actual = [point[1] for point in pr_curve.points()]
    rec_expected = [100/120, 110/120]
    prec_expected = [100/140, 110/155]

    assert prec_actual == approx(prec_expected)
    assert rec_actual == approx(rec_expected)

def test_roc_curve_reconversion():
    fpr_vals = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    tpr_vals = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250]
    points = zip(fpr_vals, tpr_vals)

    roc_curve = ROCCurve(points, 20/2000, label='Test PR Curve')
    orig_points = roc_curve.points()
    reconverted_points = roc_curve.to_pr().to_roc().points()

    assert np.array(orig_points).flatten() == approx(np.array(reconverted_points).flatten())

def test_roc_curve_reconversion_edge_cases():

    # TPR > 1.
    with pytest.raises(ValueError):
        fpr_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        tpr_vals = [1.1, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.0]
        points = zip(fpr_vals, tpr_vals)
        roc_curve = ROCCurve(points, 20/2000, label='Bad ROC Curve')

    # TPR < 0.
    with pytest.raises(ValueError):
        fpr_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        tpr_vals = [1.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, -0.01]
        points = zip(fpr_vals, tpr_vals)
        roc_curve = ROCCurve(points, 20/2000, label='Bad ROC Curve')

    # Ratio of positive to negative samples is == 0.
    with pytest.raises(ValueError):
        fpr_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        tpr_vals = [1.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.0]
        points = zip(fpr_vals, tpr_vals)
        roc_curve = ROCCurve(points, 0, label='Bad ROC Curve')

    # Ratio of positive to negative samples is < 0.
    with pytest.raises(ValueError):
        fpr_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        tpr_vals = [1.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.0]
        points = zip(fpr_vals, tpr_vals)
        roc_curve = ROCCurve(points, -10, label='Bad ROC Curve')

    fpr_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
    tpr_vals = [1.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.0]
    points = zip(fpr_vals, tpr_vals)
    roc_curve = ROCCurve(points, 20/2000, label='Good ROC Curve')

    orig_points = roc_curve.points()
    reconverted_points = roc_curve.to_pr().to_roc().points()

    assert np.array(orig_points).flatten() == approx(np.array(reconverted_points).flatten())

def test_roc_curve_resample():
    fpr_vals = [0.25, 0.5]
    tpr_vals = [0.5, 0.25]
    points = zip(fpr_vals, tpr_vals)
    roc_curve = ROCCurve(points, 20/2000, label='Test PR Curve')

    points = roc_curve.resample(num_points=6).points()
    fpr_vals_sampled = [point[0] for point in points]
    tpr_vals_sampled = [point[1] for point in points]
    fpr_vals_expected = np.linspace(0.25, 0.5, num=6)
    tpr_vals_expected = np.linspace(0.5, 0.25, num=6)

    assert fpr_vals_sampled == approx(fpr_vals_expected, abs=1e-3)
    assert tpr_vals_sampled == approx(tpr_vals_expected, abs=1e-3)

