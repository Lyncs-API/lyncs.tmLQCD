from lyncs_tmLQCD import Gauge
import numpy as np


def test_unity():
    gauge = Gauge(np.zeros((4, 4, 4, 4, 4, 3, 3), dtype="complex"))
    gauge.unity()
    assert gauge.plaquette() == 1
    assert gauge.temporal_plaquette() == 1
    assert gauge.spatial_plaquette() == 1
    assert gauge.rectangles() == 1


def test_random():
    gauge = Gauge(np.zeros((4, 4, 4, 4, 4, 3, 3), dtype="complex"))
    gauge.random()
    assert -1 <= gauge.plaquette() <= 1
    assert np.isclose(
        gauge.plaquette(), (gauge.temporal_plaquette() + gauge.spatial_plaquette()) / 2
    )
