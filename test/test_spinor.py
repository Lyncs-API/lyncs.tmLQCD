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


def test_zero():
    spinor = Spinor(np.ones((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.zero()
    assert (spinor == 0).all()


def test_unit():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    for s in range(4):
        for c in range(3):
            mat = np.zeros((4, 3), dtype="complex")
            mat[s][c] = 1
            spinor.unit(s, c)
            assert (spinor == mat).all()


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


def test_gamma5():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random()
    gamma5 = spinor.gamma5()
    same = gamma5.gamma5()
    assert np.allclose(spinor, same)


def test_proj():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random()
    pplus = spinor.proj_plus()
    zero = pplus.proj_minus()
    assert np.allclose(zero, 0)


def test_half():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    half = spinor.half()
    assert spinor.size == half.size * 2


def test_even_odd():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random()
    even = spinor.even()
    odd = spinor.odd()
    even2, odd2 = spinor.even_odd()
    assert (even == even2).all()
    assert (odd == odd2).all()

    spinor2 = np.zeros_like(spinor)
    spinor2.set_even(even)
    spinor2.set_odd(odd)
    assert (spinor == spinor2).all()

    spinor2.zero()
    spinor2.set_even_odd(even, odd)
    assert (spinor == spinor2).all()


def test_g5_even_odd():
    spinor = Spinor(np.zeros((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.random()
    even, odd = spinor.even_odd()
    even = even.gamma5()
    odd = odd.gamma5()

    spinor2 = np.zeros_like(spinor)
    spinor2.set_even_odd(even, odd)
    assert (spinor.gamma5() == spinor2).all()


def test_even_zero():
    spinor = Spinor(np.ones((4, 4, 4, 4, 4, 3), dtype="complex"))
    spinor.even_zero()
    assert (spinor.even() == 0).all()
    even, odd = spinor.even_odd()
    assert (even == 0).all()
    assert (odd == 1).all()
