import tempfile
from pickle import dumps
from pytest import raises
import numpy as np
from lyncs_tmLQCD import Gauge, gauge
from lyncs_tmLQCD.gauge import get_g_iup, get_g_gauge_field

lattice = (4, 4, 4, 4)


def test_init():
    gf = Gauge(np.zeros(lattice + (4, 3, 3), dtype="complex"))
    assert gf is Gauge(gf)
    assert gf.shape == gauge().shape
    with raises(ValueError):
        Gauge(np.zeros(lattice, dtype="complex"))
    with raises(TypeError):
        Gauge(np.zeros(lattice + (4, 3, 3), dtype="float"))


def test_unity():
    gf = gauge(lattice=lattice)
    gf.unity()
    assert gf.plaquette() == 1
    assert gf.temporal_plaquette() == 1
    assert gf.spatial_plaquette() == 1
    assert gf.rectangles() == 1
    assert gf.gauge_action() == 6 * 4 ** 4
    assert np.isclose(
        gf.symanzik_gauge_action(), 6 * 4 ** 4 * (1 + 8 / 12 - 2 * 1 / 12)
    )
    assert np.isclose(
        gf.iwasaki_gauge_action(), 6 * 4 ** 4 * (1 + 8 * 0.331 - 2 * 0.331)
    )


def test_random():
    gf = gauge(lattice=lattice)
    gf.random()
    assert -1 <= gf.plaquette() <= 1
    assert np.isclose(
        gf.plaquette(), (gf.temporal_plaquette() + gf.spatial_plaquette()) / 2
    )


def test_dumps():
    gf = gauge(lattice=lattice)
    gf.random()
    assert dumps(gf)


def test_global():
    gf = gauge(lattice=lattice)
    gf.random()
    gf.copy_to_global()
    assert (gf == get_g_gauge_field()).all()

    gf_copy = gauge(lattice=lattice)
    gf_copy.copy_from_global()
    assert (gf == gf_copy).all()


def test_io():
    tmp = tempfile.mkdtemp()
    gf = gauge(lattice=lattice)
    gf.random()
    gf.write(tmp + "/conf")
    gf_read = gauge(lattice=lattice)
    gf_read.read(tmp + "/conf")
    assert (gf == gf_read).all()


# def test_stout():
#     gf = gauge(lattice=lattice)
#     gf.random()
#     stout = gf.stout_smearing(0.1,1)
#     assert (stout==gf).all()


def test_g_iup():
    g_iup = get_g_iup()
    g_iup[0, 0, 0, 0, 0] == 1
