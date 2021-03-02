"""
Functions for spinor field
"""

__all__ = [
    "Spinor",
]

from numpy import empty_like
from lyncs_cppyy.ll import to_pointer
from lyncs_utils import static_property
from .base import Field
from .lib import lib


class _Spinor(Field):
    "class containing common methods for Spinor and half spinor"

    @static_property
    def field_shape():
        "Shape of the field"
        return (4, 3)

    @property
    def spinor(self):
        "spinor view of the field"
        return to_pointer(self.ptr, "spinor **")

    def zero(self):
        "Creates a zero field"
        lib.zero_spinor_field(self.spinor, self.volume)

    def unit(self, spin, col):
        "Creates a unitary field where all the components (spin, col) are one"
        assert 0 <= spin < 4
        assert 0 <= col < 3
        lib.constant_spinor_field(self.spinor, spin * 3 + col, self.volume)

    def gamma5(self):
        "Returns a new spinor multiplied by gamma5"
        out = empty_like(self)
        lib.gamma5(out.spinor, self.spinor, self.volume)
        return out

    def proj_plus(self):
        "Returns a new spinor with positive projection"
        out = empty_like(self)
        lib.P_plus(out.spinor, self.spinor, self.volume)
        return out

    def proj_minus(self):
        "Returns a new spinor with negative projection"
        out = empty_like(self)
        lib.P_minus(out.spinor, self.spinor, self.volume)
        return out


class Spinor(_Spinor):
    "Interface for spinor field"

    def even_zero(self):
        "Sets even sites to zero"
        lib.set_even_to_zero(self.spinor)

    def half(self):
        "Returns an empty half spinor"
        shape = list(self.shape)
        assert shape[3] % 2 == 0, "Time must be divisible by 2"
        shape[3] //= 2
        return HalfSpinor(empty_like(self, shape=shape))

    def even(self):
        "Returns the even components of the field"
        half = self.half()
        lib.convert_lexic_to_even(half.spinor, self.spinor)
        return half

    def set_even(self, half):
        "Gets the components of the given field into the even sites"
        lib.convert_even_to_lexic(self.spinor, HalfSpinor(half).spinor)

    def odd(self):
        "Returns the odd components of the field"
        half = self.half()
        lib.convert_lexic_to_odd(half.spinor, self.spinor)
        return half

    def set_odd(self, half):
        "Gets the components of the given field into the odd sites"
        lib.convert_odd_to_lexic(self.spinor, HalfSpinor(half).spinor)

    def even_odd(self):
        "Returns the even and odd components of the field"
        even = self.half()
        odd = self.half()
        lib.convert_lexic_to_eo(even.spinor, odd.spinor, self.spinor)
        return (even, odd)

    def set_even_odd(self, even, odd):
        "Gets the components of the given field into the odd sites"
        lib.convert_eo_to_lexic(
            self.spinor, HalfSpinor(even).spinor, HalfSpinor(odd).spinor
        )

    def random(self, repro=False):
        "Creates a uniform in [0,1] random field"
        lib.random_spinor_field_lexic(self.spinor, repro, lib.RN_UNIF)

    def random_pm1(self, repro=False):
        "Creates a uniform in [-1,1] random field"
        lib.random_spinor_field_lexic(self.spinor, repro, lib.RN_PM1UNIF)

    def random_gauss(self, repro=False):
        "Creates a Gaussian random field with zero mean value and 1 standard deviation"
        lib.random_spinor_field_lexic(self.spinor, repro, lib.RN_GAUSS)

    def random_Z2(self, repro=False):
        "Creates a Z2 random field containing square roots of 1"
        lib.random_spinor_field_lexic(self.spinor, repro, lib.RN_Z2)


class HalfSpinor(_Spinor):
    @property
    def volume(self):
        "The total lattice volume"
        return super().volume // 2

    @property
    def lattice(self):
        "The total lattice volume"
        return self.shape[:3] + (self.shape[3] * 2,)
