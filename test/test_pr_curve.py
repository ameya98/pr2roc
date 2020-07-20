from __future__ import division

from roc2pr import PRCurve

import numpy as np
import pytest
from pytest import approx

def test_pr_curve_conversion():
    rec_vals = [100/120, 110/120]
    prec_vals = [100/140, 110/155]
    points = zip(rec_vals, prec_vals)
    pr_curve = PRCurve(points, 120/60, label='Test PR Curve')

    roc_curve = pr_curve.to_roc()
    fpr_actual = [point[0] for point in roc_curve.points()]
    tpr_actual = [point[1] for point in roc_curve.points()]
    fpr_expected = [40/60, 45/60]
    tpr_expected = [100/120, 110/120]

    assert tpr_actual == approx(tpr_expected)
    assert fpr_actual == approx(fpr_expected)

def test_pr_curve_reconversion():
    rec_vals = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    prec_vals = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250]
    points = zip(rec_vals, prec_vals)

    pr_curve = PRCurve(points, 20/2000, label='Test PR Curve')
    orig_points = pr_curve.points()
    reconverted_points = pr_curve.to_roc().to_pr().points()

    assert np.array(orig_points).flatten() == approx(np.array(reconverted_points).flatten())

def test_pr_curve_reconversion_edge_cases():

    # Precision > 1.
    with pytest.raises(ValueError):
        rec_vals = [0.1, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [1.1, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Precision < 0.
    with pytest.raises(ValueError):
        rec_vals = [0.1, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [1.1, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, -0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Recall > 1.
    with pytest.raises(ValueError):
        rec_vals = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.1]
        prec_vals = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Recall < 0.
    with pytest.raises(ValueError):
        rec_vals = [-0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Ratio of positive to negative samples is == 0.
    with pytest.raises(ValueError):
        rec_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [0.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 0, label='Bad PR Curve')

    # Ratio of positive to negative samples is < 0.
    with pytest.raises(ValueError):
        rec_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [0.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, -10, label='Bad PR Curve')

    # Precision > 0 when Recall = 0.
    with pytest.raises(ValueError):
        rec_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [0.9, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Precision = 0 when Recall > 0.
    with pytest.raises(ValueError):
        rec_vals = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
        prec_vals = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.0]
        points = zip(rec_vals, prec_vals)
        pr_curve = PRCurve(points, 20/2000, label='Bad PR Curve')

    # Precision = 0 when Recall = 0. This is acceptable.
    rec_vals = [0.0, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 1.0]
    prec_vals = [0.0, 0.5, 0.375, 0.318, 0.286, 0.265, 0.250, 0.01]
    points = zip(rec_vals, prec_vals)
    pr_curve = PRCurve(points, 20/2000, label='Good PR Curve')

    orig_points = pr_curve.points()
    reconverted_points = pr_curve.to_roc().to_pr().points()

    assert np.array(orig_points).flatten() == approx(np.array(reconverted_points).flatten())

def test_pr_curve_resample():
    rec_vals = [0.25, 0.5]
    prec_vals = [0.5, 0.25]
    points = zip(rec_vals, prec_vals)
    pr_curve = PRCurve(points, 20/2000, label='Test PR Curve')

    points = pr_curve.resample(num_points=6).points()
    rec_vals_sampled = [point[0] for point in points]
    prec_vals_sampled = [point[1] for point in points]
    rec_vals_expected = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    prec_vals_expected = [0.5, 0.375, 0.318, 0.286, 0.265, 0.250]

    assert rec_vals_sampled == approx(rec_vals_expected, abs=1e-3)
    assert prec_vals_sampled == approx(prec_vals_expected, abs=1e-3)

