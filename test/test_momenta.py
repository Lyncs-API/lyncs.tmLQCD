import numpy as np
from pytest import raises
from lyncs_tmLQCD import Momenta


def test_init():
    momenta = Momenta(np.zeros((4, 4, 4, 4, 4, 8), dtype="double"))
    assert (momenta == Momenta(momenta)).all()
    with raises(ValueError):
        Momenta(np.zeros((4, 4, 4, 4), dtype="double"))
    with raises(TypeError):
        Momenta(np.zeros((4, 4, 4, 4, 4, 8), dtype="complex"))


def test_random():
    momenta = Momenta(np.zeros((4, 4, 4, 4, 4, 8), dtype="double"))
    momenta.random()
    assert np.isclose(momenta.mean(), 0.0, atol=0.1)
    assert np.isclose(momenta.std(), 1.0, atol=0.1)
