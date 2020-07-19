from __future__ import division

from roc2pr import Curve

import numpy as np
import pytest
from pytest import approx

def check_curve(curve):
    assert curve is not None

    assert curve.points()[0] == (0, 1)
    assert curve.points()[4] == (8, 3)

    resampled_curve = curve.resample(2)
    assert resampled_curve.points()[0] == approx((0, 1))
    assert resampled_curve.points()[1] == approx((8, 3))

    resampled_curve = curve.resample(3)
    assert resampled_curve.points()[0] == approx((0, 1))
    assert resampled_curve.points()[1] == approx((4, 9.5))
    assert resampled_curve.points()[2] == approx((8, 3))

    resampled_curve = curve.resample(4)
    assert resampled_curve.points()[0] == approx((0, 1))
    assert resampled_curve.points()[1] == approx((8/3, 49/6))
    assert resampled_curve.points()[2] == approx((16/3, 83/9))
    assert resampled_curve.points()[3] == approx((8, 3))

    resampled_curve = curve.resample(5)
    assert resampled_curve.points()[0] == approx((0, 1))
    assert resampled_curve.points()[1] == approx((2, 13/2))
    assert resampled_curve.points()[2] == approx((4, 19/2))
    assert resampled_curve.points()[3] == approx((6, 23/3))
    assert resampled_curve.points()[4] == approx((8, 3))

def test_sorted():
    x_vals = [0, 1, 3, 5, 8]
    y_vals = [1, 4, 9, 10, 3]
    points = zip(x_vals, y_vals)

    curve = Curve(points, label='Test Curve')
    check_curve(curve)

def test_unsorted():
    x_vals = [8, 1, 5, 3, 0]
    y_vals = [3, 4, 10, 9, 1]
    points = zip(x_vals, y_vals)
    
    curve = Curve(points, label='Test Curve')
    check_curve(curve)

def test_np_array():
    x_vals = [8, 1, 5, 3, 0]
    y_vals = [3, 4, 10, 9, 1]
    points = np.zeros((5, 2))
    points[:, 0] = x_vals
    points[:, 1] = y_vals
    
    curve = Curve(points, label='Test Curve')
    check_curve(curve)

def test_bad_curve():
    with pytest.raises(ValueError):
        points = [(0, 1)]
        curve = Curve(points, label='Bad Curve')

    with pytest.raises(ValueError):
        points = [0, 1, 2]
        curve = Curve(points, label='Bad Curve')

    with pytest.raises(ValueError):
        points = 1
        curve = Curve(points, label='Bad Curve')
