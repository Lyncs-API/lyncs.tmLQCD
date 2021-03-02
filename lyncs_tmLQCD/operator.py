__all__ = [
    "Operator",
]

from dataclasses import dataclass
from functools import wraps
from .gauge import Gauge
from .lib import lib


@dataclass
class Operator:
    gauge: Gauge
    kappa: float = 0.125
    mu: float = 0.0
    csw: float = 0.0
    eps: float = 0.0
    half: bool = False

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
        assert lib.no_operators < lib.max_no_operators

        otype = lib.WILSON
        if self.mu != 0:
            otype = lib.TMWILSON
        if self.csw != 0:
            otype = lib.CLOVER
        if doublet:
            otype = lib.DBTMWILSON
            if self.csw != 0:
                otype = lib.DBCLOVER

        lib.add_operator(otype)
        ope = lib.operator_list[lib.no_operators - 1]
        ope.even_odd_flag = 1 if self.half else 0
        ope.kappa = self.kappa
        ope.mu = self.mu
        ope.mubar = self.mu
        ope.epsbar = self.eps
        ope.c_sw = self.csw

        lib.init_operators()
        fnc = getattr(ope, f"apply{name}")
        lib.no_operators -= 1

        # INIT
        self.gauge.copy_to_global()
        # TODO: initialize sw_term
        pass

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
    def DbQsq(self):
        """
        Q^2(mu,eps) doublet operator acting on the full vector or
        (if half) acting on the odd part of an even-odd reduced vector
        """
        if not self.half:
            raise ValueError("DbQsq can be used only on even-odd reduced vectors")
        return self.get_operator("DbQsq", doublet=True)


Gauge.operator = wraps(Operator)(lambda self, **kwargs: Operator(self, **kwargs))
