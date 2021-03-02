import numpy as np
from lyncs_tmLQCD import Gauge


def test_init():
    gauge = Gauge(np.zeros((4, 4, 4, 4, 4, 3, 3), dtype="complex"))
    gauge.operator()
