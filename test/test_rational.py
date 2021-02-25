from lyncs_tmLQCD import ellipticK, ellipticKm1, sncndn, zolotarev
from scipy.special import ellipk, ellipkm1, ellipj
from numpy import isclose, allclose, linspace, prod


def test_ellipticK():
    for k in linspace(0, 0.9):
        assert isclose(ellipticK(k), ellipk(k))
    for k in linspace(1, 10):
        assert isclose(ellipticK(1 - 10 ** -k), ellipk(1 - 10 ** -k))
    for k in linspace(0.1, 1):
        assert isclose(ellipticKm1(k), ellipkm1(k))
    for k in linspace(1, 10):
        assert isclose(ellipticKm1(10 ** -k), ellipkm1(10 ** -k))


def test_sncndn():
    for u in linspace(-10, 10):
        for k in linspace(0, 0.9):
            assert allclose(sncndn(u, k), ellipj(u, k)[:-1])


def test_zolotarev():
    for n in range(1, 10):
        A, num, den, delta = zolotarev(n, 0.1)
        for x in linspace(0.1, 1):
            y = A * prod(x + num) / prod(x + den)
            assert isclose(y, 1 / x ** 0.5, rtol=delta)
