from __future__ import division

from __future__ import division

from .curve import Curve
from numpy import min, max, seterr
seterr(all='raise')

class ROCCurve(Curve):
    def __init__(self, points, pos_neg_ratio, label=None):
        Curve.__init__(self, points, label)
        self.pos_neg_ratio = pos_neg_ratio

        if max([self.x_vals, self.y_vals]) > 1:
            raise ValueError('TPR and FPR cannot be greater than 1.')
        if min([self.x_vals, self.y_vals]) < 0:
            raise ValueError('TPR and FPR cannot be lesser than 0.')
        if self.pos_neg_ratio <= 0:
            raise ValueError('\'pos_neg_ratio\' must be > 0.')

    def compute_precision_vals(self):
        def compute_precision_val(fpr, tpr):
            try:
                return 1/(1 + (fpr/tpr) * 1/(self.pos_neg_ratio))
            except (ZeroDivisionError, FloatingPointError):
                return 0

        return [compute_precision_val(fpr, tpr) for fpr, tpr in zip(self.x_vals, self.y_vals)]

    def to_pr(self):
        from .pr_curve import PRCurve

        recall_vals = self.y_vals
        precision_vals = self.compute_precision_vals()

        points = zip(recall_vals, precision_vals)
        return PRCurve(points, self.pos_neg_ratio)

    def resample(self, num_points):
        resampled_curve = Curve.resample(self, num_points)
        return ROCCurve(resampled_curve.points(), pos_neg_ratio=self.pos_neg_ratio)