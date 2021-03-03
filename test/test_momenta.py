import numpy as np
from pytest import raises
from lyncs_tmLQCD import Momenta, momenta

lattice = (4, 4, 4, 4)


def test_init():
    mom = Momenta(np.zeros(lattice + (4, 8), dtype="double"))
    assert mom is Momenta(mom)
    assert mom.shape == momenta().shape
    with raises(ValueError):
        Momenta(np.zeros(lattice, dtype="double"))
    with raises(TypeError):
        Momenta(np.zeros(lattice + (4, 8), dtype="complex"))


def test_random():
    mom = momenta(lattice=lattice)
    mom.random()
    assert np.isclose(mom.mean(), 0.0, atol=0.1)
    assert np.isclose(mom.std(), 1.0, atol=0.1)
