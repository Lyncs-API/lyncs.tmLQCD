"""
Functions for spinor field
"""

__all__ = [
    "Spinor",
]

from lyncs_cppyy.ll import to_pointer
from lyncs_utils import static_property
from .base import Field
from .lib import lib


class Spinor(Field):
    "Interface for spinor fields"

    @static_property
    def field_shape():
        "Shape of the field"
        return (4, 3)

    @property
    def spinor(self):
        "spinor view of the field"
        return to_pointer(self.ptr, "spinor **")

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
