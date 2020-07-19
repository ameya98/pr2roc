try:
    from .curve import Curve
    from .pr_curve import PRCurve
    from .roc_curve import ROCCurve
except ImportError:
    from curve import Curve
    from pr_curve import PRCurve
    from roc_curve import ROCCurve
