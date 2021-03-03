import numpy as np
from lyncs_tmLQCD import Gauge

half_operators = ["M", "Q", "Mee", "MeeInv", "Qp", "Qm", "Qsq", "Mp", "Mm", "DbQsq"]
full_operators = ["Qp", "Qm", "Qsq", "Mp", "Mm"]
doublet_operators = ["M", "Q", "DbQsq"]


def test_init():
    gauge = Gauge(np.zeros((4, 4, 4, 4, 4, 3, 3), dtype="complex"))

    ope = gauge.operator(csw=0, half=False)
    for name in full_operators:
        assert callable(getattr(ope, name))

    ope = gauge.operator(csw=1, half=False)
    for name in full_operators:
        assert callable(getattr(ope, name))

    ope = gauge.operator(csw=0, half=True)
    for name in half_operators:
        assert callable(getattr(ope, name))

    ope = gauge.operator(csw=1, half=True)
    for name in half_operators:
        assert callable(getattr(ope, name))
