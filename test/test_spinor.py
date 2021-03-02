import numpy as np
from pytest import raises
from lyncs_tmLQCD import Spinor


def test_init():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    assert (spinor == Spinor(spinor)).all()
    with raises(ValueError):
        Spinor(np.zeros((4, 4, 4, 4), dtype="complex"))
    with raises(TypeError):
        Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="float"))


def test_random():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random()
    mean = spinor.mean()
    assert np.isclose(mean.real, 0.5, atol=0.1)
    assert np.isclose(mean.imag, 0.5, atol=0.1)
    assert np.isclose(spinor.std(), 0.4, atol=0.1)


def test_random_pm1():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random_pm1()
    mean = spinor.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spinor.std(), 0.81, atol=0.1)


def test_gauss():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random_gauss()
    mean = spinor.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spinor.std(), 1, atol=0.1)


def test_Z2():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random_Z2()
    mean = spinor.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spinor.std(), 1, atol=0.1)
    assert np.allclose(spinor.conj() * spinor, 1)
