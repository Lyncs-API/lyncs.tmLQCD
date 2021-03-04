from pytest import raises
import numpy as np
from lyncs_tmLQCD import gauge, spinor, half_spinor

lattice = (4, 4, 4, 4)
half_operators = ["M", "Q", "Mee", "MeeInv", "Qp", "Qm", "Qsq", "Mp", "Mm", "DbQsq"]
full_operators = ["Qp", "Qm", "Qsq", "Mp", "Mm"]
doublet_operators = ["M", "Q", "DbQsq"]


def test_init():
    gf = gauge(lattice=lattice)

    ope = gf.operator(csw=0, half=False)
    for name in full_operators:
        assert callable(getattr(ope, name))

    ope = gf.operator(csw=1, half=False)
    for name in full_operators:
        assert callable(getattr(ope, name))

    ope = gf.operator(csw=0, half=True)
    for name in half_operators:
        assert callable(getattr(ope, name))

    ope = gf.operator(csw=1, half=True)
    for name in half_operators:
        assert callable(getattr(ope, name))

    ope = gf.operator(csw=0, half=False)
    for name in half_operators:
        if name not in full_operators:
            with raises(ValueError):
                getattr(ope, name)


def test_apply():
    gf = gauge(lattice=lattice)
    spin = spinor(lattice=lattice)
    gf.random()
    spin.random()
    even = spin.even()
    odd = spin.odd()

    ope = gf.operator(csw=0, half=False)
    for name in full_operators:
        out = getattr(ope, name)(spin)
        assert out.shape == spin.shape

    ope = gf.operator(csw=1, half=False)
    for name in full_operators:
        out = getattr(ope, name)(spin)
        assert out.shape == spin.shape

    ope = gf.operator(csw=0, half=True)
    for name in half_operators:
        if name in doublet_operators:
            out1, out2 = getattr(ope, name)(even, odd)
            assert out1.shape == even.shape
            assert out2.shape == odd.shape
        else:
            out = getattr(ope, name)(even)
            assert out.shape == even.shape

    ope = gf.operator(csw=1, half=True)
    for name in half_operators:
        if name in doublet_operators:
            out1, out2 = getattr(ope, name)(even, odd)
            assert out1.shape == even.shape
            assert out2.shape == odd.shape
        else:
            out = getattr(ope, name)(even)
            assert out.shape == even.shape
