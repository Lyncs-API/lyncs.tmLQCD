"""
Utils for the rational approximation using Zolotarev optimal solution
"""

__all__ = [
    "ellipticK",
    "ellipticKm1",
    "sncndn",
    "zolotarev",
]

import numpy
from .lib import lib


def _rk(k):
    return (k / (1 - k)) ** 0.5


def _rkm1(k):
    return ((1 - k) / k) ** 0.5


def ellipticK(k):
    """
    Returns the complete elliptic integral K(k) for 0<=k<1.
    The parameter k is compatible with scipy.spacial.ellipk.
    """
    return lib.ellipticK(_rk(k))


def ellipticKm1(k):
    """
    Returns the complete elliptic integral `K(1-k)` for `0<=k<1`.
    The parameter `k` is compatible with `scipy.spacial.ellipkm1`.
    """
    return lib.ellipticK(_rkm1(k))


def sncndn(u, k):
    """
    Computes the Jacobi elliptic functions `sn(u,k)`, `cn(u,k)`, `dn(u,k)`
    for specified real `u` and `0<=k<1`.
    """
    out = numpy.zeros(3, dtype="float64")
    lib.sncndn(u, _rk(k), out, out[1:], out[2:])
    return tuple(out)


def zolotarev(n, eps):
    """
    Computes the amplitude, the coefficients and the error of
    the Zolotarev optimal rational approximation of degree [n,n]
    to the function f(y)=1/sqrt(y) in the range eps<=y<=1.

    Returns A, num, den, delta such that
    f(y) = A*P(y)/Q(y) = 1/sqrt(y) + O(delta), with
    P(y) = (y+num[0])*(y+num[1])*...*(y+num[n-1]),
    Q(y) = (y+den[0])*(y+den[1])*...*(y+den[n-1]).
    """
    ampl = numpy.zeros(1, dtype="float64")
    delta = numpy.zeros(1, dtype="float64")
    coeffs = numpy.zeros((n, 2), dtype="float64")
    lib.zolotarev(n, eps, ampl, coeffs, delta)
    return ampl[0], coeffs[:, 0], coeffs[:, 1], delta[0]
