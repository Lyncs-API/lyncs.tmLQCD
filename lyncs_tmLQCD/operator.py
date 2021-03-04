__all__ = [
    "Operator",
]

from dataclasses import dataclass
from functools import wraps, partial
from numpy import empty_like
from .gauge import Gauge
from .spinor import Spinor, HalfSpinor
from .lib import lib


@dataclass
class Operator:
    gauge: Gauge
    kappa: float = 0.125
    mu: float = 0.0
    csw: float = 0.0
    eps: float = 0.0
    half: bool = False

    def set_globals(self):
        "Sets the global values"
        lib.g_kappa = self.kappa
        lib.g_mu = self.mu
        lib.g_c_sw = self.csw
        lib.g_mubar = self.mu
        lib.g_epsbar = self.eps
        lib.g_update_gauge_copy = 1
        self.gauge.copy_to_global()
        if self.csw != 0:
            lib.init_sw_fields()
            lib.sw_term(self.gauge.su3_field, self.kappa, self.csw)

    def get_tmLQCD_function(self, name):
        "Returns the low-level tmLQCD function"
        if self.half:
            if self.csw == 0:
                names = {
                    "M": "M_full",
                    "Q": "Q_full",
                    "Mee": "Mee_psi",
                    "MeeInv": "Mee_inv_psi",
                    "Qp": "Qtm_plus_psi",
                    "Qm": "Qtm_minus_psi",
                    "Qsq": "Qtm_pm_psi",
                    "Mp": "Mtm_plus_psi",
                    "Mm": "Mtm_minus_psi",
                    "DbQsq": "Qtm_pm_ndpsi",
                }
            else:
                names = {
                    "M": "Msw_full",
                    "Q": "Qsw_full",
                    "Mee": "Mee_sw_psi",
                    "MeeInv": "Mee_sw_inv_psi",
                    "Qp": "Qsw_plus_psi",
                    "Qm": "Qsw_minus_psi",
                    "Qsq": "Qsw_pm_psi",
                    "Mp": "Msw_plus_psi",
                    "Mm": "Msw_minus_psi",
                    "DbQsq": "Qsw_pm_ndpsi",
                }
        else:
            if self.csw == 0:
                names = {
                    "Qp": "Q_plus_psi",
                    "Qm": "Q_minus_psi",
                    "Qsq": "Q_pm_psi",
                    "Mp": "D_psi",
                    "Mm": "M_minus_psi",
                }
            else:
                names = {
                    "Qp": "Qsw_full_plus_psi",
                    "Qm": "Qsw_full_minus_psi",
                    "Qsq": "Qsw_full_pm_psi",
                    "Mp": "D_psi",
                    "Mm": "Msw_full_minus_psi",
                }
        return getattr(lib, names[name])

    def get_operator(self, name, doublet=False):
        """
        Prepares the environment for calling the requested operator and
        returns a function to be used for the call. Note that the function
        will applied correctly as far as the global parameters of tmLQCD,
        as the gauge field, kappa, mu etc, stay unchanged.

        To avoid overheads, e.g. during the inversion of the operator,
        the returned function can be stored and used multiple times.
        If instead various operators need to be called, then a new function
        needs to be instantiated every time.
        """

        fnc = self.get_tmLQCD_function(name)
        self.set_globals()
        spinor = lambda sp: HalfSpinor(sp) if self.half else Spinor(sp)
        if name.startswith("Mee"):
            fnc = partial(fnc, mu=self.mu)

        if doublet:

            def caller(even, odd, out_even=None, out_odd=None):
                even = spinor(even)
                odd = spinor(odd)
                out_even = empty_like(even) if out_even is None else spinor(out_even)
                out_odd = empty_like(odd) if out_odd is None else spinor(out_odd)
                fnc(out_even.spinor, out_odd.spinor, even.spinor, odd.spinor)
                return (out_even, out_odd)

        else:

            def caller(vec, out_vec=None):
                vec = spinor(vec)
                out_vec = empty_like(vec) if out_vec is None else spinor(out_vec)
                fnc(out_vec.spinor, vec.spinor)
                return out_vec

        caller.__name__ = name
        caller.__doc__ = getattr(Operator, name).__doc__
        return caller

    @property
    def Mee(self):
        "even-even part of the even-odd operator"
        if not self.half:
            raise ValueError("Mee can be used only on even-odd reduced vectors")
        return self.get_operator("Mee")

    @property
    def MeeInv(self):
        "inverse of the even-even part of the even-odd operator"
        if not self.half:
            raise ValueError("MeeInv can be used only on even-odd reduced vectors")
        return self.get_operator("MeeInv")

    @property
    def Mp(self):
        """
        M(+mu) operator acting on the full vector or (if half)
        acting on the odd part of an even-odd reduced vector
        """
        return self.get_operator("Mp")

    @property
    def Mm(self):
        """
        M(-mu) operator acting on the full vector or (if half)
        acting on the odd part of an even-odd reduced vector
        """
        return self.get_operator("Mm")

    @property
    def Qp(self):
        """
        Q(+mu)=g5*M(+mu) operator acting on the full vector or (if half)
        acting on the odd part of an even-odd reduced vector
        """
        return self.get_operator("Qp")

    @property
    def Qm(self):
        """
        Q(-mu)=g5*M(-mu) operator acting on the full vector or (if half)
        acting on the odd part of an even-odd reduced vector
        """
        return self.get_operator("Qm")

    @property
    def Qsq(self):
        """
        Q^2=Q(+mu)*Q(-mu) operator acting on the full vector or (if half)
        acting on the odd part of an even-odd reduced vector
        """
        return self.get_operator("Qsq")

    @property
    def M(self):
        """
        M(mu) operator acting on both even and odd part of an even-odd reduced vector
        """
        if not self.half:
            raise ValueError("M can be used only on even-odd reduced vectors")
        return self.get_operator("M", doublet=True)

    @property
    def Q(self):
        """
        Q(mu)=g5*M(mu) operator acting on both even and odd part of an even-odd reduced vector
        """
        if not self.half:
            raise ValueError("Q can be used only on even-odd reduced vectors")
        return self.get_operator("Q", doublet=True)

    @property
    def DbQsq(self):
        """
        Q^2(mu,eps) doublet operator acting on the odd part of an even-odd reduced vector
        """
        if not self.half:
            raise ValueError("DbQsq can be used only on even-odd reduced vectors")
        return self.get_operator("DbQsq", doublet=True)


Gauge.operator = wraps(Operator)(lambda self, **kwargs: Operator(self, **kwargs))
