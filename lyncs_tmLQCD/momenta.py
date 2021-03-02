"""
Functions for momenta field
"""

__all__ = [
    "Momenta",
]

from lyncs_cppyy.ll import to_pointer
from lyncs_utils import static_property
from .base import Field
from .lib import lib


class Momenta(Field):
    "Interface for spinor fields"

    @static_property
    def field_shape():
        "Shape of the field"
        return (4, 8)

    @static_property
    def field_dtype():
        "Data type of the field"
        return "double"

    @property
    def su3adj(self):
        return to_pointer(self.ptr, "su3adj **")

    def random(self, repro=False):
        "Creates a random momenta field"
        lib.random_su3adj_field(repro, self.su3adj)
