import numpy as np
from pytest import raises
from lyncs_tmLQCD import Spinor, spinor, half_spinor

lattice = (4, 4, 4, 4)


def test_init():
    spin = Spinor(np.zeros(lattice + (4, 3), dtype="complex"))
    assert spin is Spinor(spin)
    assert spin.shape == spinor().shape
    with raises(ValueError):
        Spinor(np.zeros(lattice, dtype="complex"))
    with raises(TypeError):
        Spinor(np.zeros(lattice + (4, 3), dtype="float"))


def test_zero():
    spin = spinor(lattice=lattice)
    spin.zero()
    assert (spin == 0).all()


def test_unit():
    spin = spinor(lattice=lattice)
    for s in range(4):
        for c in range(3):
            mat = np.zeros((4, 3), dtype="complex")
            mat[s][c] = 1
            spin.unit(s, c)
            assert (spin == mat).all()


def test_random():
    spin = spinor(lattice=lattice)
    spin.random()
    mean = spin.mean()
    assert np.isclose(mean.real, 0.5, atol=0.1)
    assert np.isclose(mean.imag, 0.5, atol=0.1)
    assert np.isclose(spin.std(), 0.4, atol=0.1)


def test_random_pm1():
    spin = spinor(lattice=lattice)
    spin.random_pm1()
    mean = spin.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spin.std(), 0.81, atol=0.1)


def test_gauss():
    spin = spinor(lattice=lattice)
    spin.random_gauss()
    mean = spin.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spin.std(), 1, atol=0.1)


def test_Z2():
    spin = spinor(lattice=lattice)
    spin.random_Z2()
    mean = spin.mean()
    assert np.isclose(mean.real, 0.0, atol=0.1)
    assert np.isclose(mean.imag, 0.0, atol=0.1)
    assert np.isclose(spin.std(), 1, atol=0.1)
    assert np.allclose(spin.conj() * spin, 1)


def test_gamma5():
    spin = spinor(lattice=lattice)
    spin.random()
    gamma5 = spin.gamma5()
    same = gamma5.gamma5()
    assert np.allclose(spin, same)


def test_proj():
    spin = spinor(lattice=lattice)
    spin.random()
    pplus = spin.proj_plus()
    zero = pplus.proj_minus()
    assert np.allclose(zero, 0)


def test_half():
    spin = spinor(lattice=lattice)
    half = spin.half()
    assert spin.size == half.size * 2
    assert half.shape == half_spinor(lattice=lattice).shape
    assert half.shape == half_spinor().shape


def test_even_odd():
    spin = spinor(lattice=lattice)
    spin.random()
    even = spin.even()
    odd = spin.odd()
    even2, odd2 = spin.even_odd()
    assert (even == even2).all()
    assert (odd == odd2).all()

    spin2 = np.zeros_like(spin)
    spin2.set_even(even)
    spin2.set_odd(odd)
    assert (spin == spin2).all()

    spin2.zero()
    spin2.set_even_odd(even, odd)
    assert (spin == spin2).all()


def test_g5_even_odd():
    spin = spinor(lattice=lattice)
    spin.random()
    even, odd = spin.even_odd()
    even = even.gamma5()
    odd = odd.gamma5()

    spin2 = np.zeros_like(spin)
    spin2.set_even_odd(even, odd)
    assert (spin.gamma5() == spin2).all()


def test_even_zero():
    spin = spinor(lattice=lattice)
    spin[:] = 1
    spin.even_zero()
    assert (spin.even() == 0).all()
    even, odd = spin.even_odd()
    assert (even == 0).all()
    assert (odd == 1).all()
