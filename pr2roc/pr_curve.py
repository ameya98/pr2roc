from __future__ import division

from .curve import Curve
from numpy import min, max, seterr
seterr(all='raise')

class PRCurve(Curve):
    def __init__(self, points, pos_neg_ratio, label=None):
        Curve.__init__(self, points, label)
        self.pos_neg_ratio = pos_neg_ratio

        if max([self.x_vals, self.y_vals]) > 1:
            raise ValueError('Precision and recall cannot be greater than 1.')
        if min([self.x_vals, self.y_vals]) < 0:
            raise ValueError('Precision and recall cannot be lesser than 0.')
        if self.pos_neg_ratio <= 0:
            raise ValueError('\'pos_neg_ratio\' must be >= 0.')

        for x, y in zip(self.x_vals, self.y_vals):
            if x > 0 and y == 0:
                raise ValueError('Precision cannot be 0 if recall is > 0.')

            if x == 0 and y > 0:
                raise ValueError('Precision cannot be > 0 if recall is 0. %s %s' % (self.x_vals, self.y_vals))

    def compute_fpr_vals(self):
        def compute_fpr_val(rec, prec):
            try:
                return rec * self.pos_neg_ratio * (1/prec - 1)
            except (ZeroDivisionError, FloatingPointError):
                return 1

        return [compute_fpr_val(rec, prec) for rec, prec in zip(self.x_vals, self.y_vals)]

    def to_roc(self):
        from .roc_curve import ROCCurve

        fpr_vals = self.compute_fpr_vals()
        tpr_vals = self.x_vals

        points = zip(fpr_vals, tpr_vals)
        return ROCCurve(points, self.pos_neg_ratio)

    def resample(self, num_points):
        return self.to_roc().resample(num_points).to_pr()