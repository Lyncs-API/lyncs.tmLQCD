from pytest import raises
import numpy as np
from lyncs_tmLQCD import gauge, spinor, half_spinor

lattice = (4, 4, 4, 4)
half_operators = ["M", "Q", "Mee", "MeeInv", "Qp", "Qm", "Qsq", "Mp", "Mm", "DbQsq"]
full_operators = ["Qp", "Qm", "Qsq", "Mp", "Mm"]
doublet_operators = ["M", "Q", "DbQsq"]


def test_init():
    gf = gauge(lattice=lattice)

    ops = gf.operator(csw=0, half=False)
    for name in full_operators:
        assert callable(getattr(ops, name))

    ops = gf.operator(csw=1, half=False)
    for name in full_operators:
        assert callable(getattr(ops, name))

    ops = gf.operator(csw=0, half=True)
    for name in half_operators:
        assert callable(getattr(ops, name))

    ops = gf.operator(csw=1, half=True)
    for name in half_operators:
        assert callable(getattr(ops, name))

    ops = gf.operator(csw=0, half=False)
    for name in half_operators:
        if name not in full_operators:
            with raises(ValueError):
                getattr(ops, name)


def test_apply():
    gf = gauge(lattice=lattice)
    spin = spinor(lattice=lattice)
    gf.random()
    spin.random()
    even = spin.even()
    odd = spin.odd()

    ops = gf.operator(csw=0, half=False)
    for name in full_operators:
        out = getattr(ops, name)(spin)
        assert out.shape == spin.shape

    ops = gf.operator(csw=1, half=False)
    for name in full_operators:
        out = getattr(ops, name)(spin)
        assert out.shape == spin.shape

    ops = gf.operator(csw=0, half=True)
    for name in half_operators:
        if name in doublet_operators:
            out1, out2 = getattr(ops, name)(even, odd)
            assert out1.shape == even.shape
            assert out2.shape == odd.shape
        else:
            out = getattr(ops, name)(even)
            assert out.shape == even.shape

    ops = gf.operator(csw=1, half=True)
    for name in half_operators:
        if name in doublet_operators:
            out1, out2 = getattr(ops, name)(even, odd)
            assert out1.shape == even.shape
            assert out2.shape == odd.shape
        else:
            out = getattr(ops, name)(even)
            assert out.shape == even.shape

def test_M():
    gf = gauge(lattice=lattice)
    spin = spinor(lattice=lattice)
    gf.random()
    spin.random()
    even = spin.even()
    odd = spin.odd()
    kappa = 0.13
    csw = 1

    ops = gf.operator(csw=0, half=False)
    out = ops.Mp(spin)
    assert np.allclose(out, spin)
    
    ops = gf.operator(kappa = kappa, csw=0, half=False)
    out = ops.Mp(spin)

    ops = gf.operator(kappa = kappa, csw=0, half=True)
    oute, outo = ops.M(even, odd)

    assert np.allclose(out.even(),oute)
    assert np.allclose(out.odd(),outo)

    ops = gf.operator(kappa = kappa, csw=csw, half=False)
    ops.Mp(spin, out)

    ops = gf.operator(kappa = kappa, csw=csw, half=True)
    oute, outo = ops.M(even, odd)

    assert np.allclose(out.even(),oute)
    assert np.allclose(out.odd(),outo)
