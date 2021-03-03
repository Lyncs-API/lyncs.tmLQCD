"""
Base class for fields in tmLQCD
"""

__all__ = [
    "Field",
    "field",
]

from functools import wraps
from numpy import array, prod, ndarray, asarray, empty
from lyncs_cppyy.ll import array_to_pointers
from lyncs_utils import static_property
from .lib import lib


class Field(ndarray):
    "Interface for gauge fields"

    @static_property
    def field_shape():
        "Shape of the field"
        raise NotImplementedError("Method to be overwritten")

    @static_property
    def field_dtype():
        "Data type of the field"
        return "complex128"

    @property
    def lattice(self):
        "Returns the lattice size"
        return self.shape[:4]

    def _check(self):
        fshape = self.field_shape
        if len(self.shape) != 4 + len(fshape) or self.shape[-len(fshape) :] != fshape:
            raise ValueError(f"Array must have shape (X,Y,Z,T,{str(fshape)[1:-1]})")
        if self.dtype != self.field_dtype:
            raise TypeError("Expected a complex field")
        lib.initialize(*self.lattice)
        return self

    def __new__(cls, input_array):
        if isinstance(input_array, cls):
            return input_array
        return asarray(input_array).view(cls)._check()

    @property
    def ptr(self):
        "Returns the raw pointer to the field"
        if not hasattr(self, "_pointers"):
            self._pointers = array_to_pointers(
                self._check().reshape((-1, prod(self.field_shape)))
            )
        return self._pointers.ptr

    @property
    def volume(self):
        "The total lattice volume"
        return lib.VOLUME


def field(array: ndarray = None, lattice: tuple = None, ftype: Field = Field):
    """
    Instantiates a new field of the given type.

    Parameters:
    -----------
    - array: array to initialize as field
    - lattice: shape of the local lattice
    - ftype: field type to be used
    """
    if array:
        return ftype(array)
    if not lattice:
        lattice = lib.lattice
    shape = lattice + ftype.field_shape
    dtype = ftype.field_dtype
    array = empty(shape=shape, dtype=dtype)
    return ftype(array)


def to_half_lattice(shape):
    "Halves the given shape"
    shape = list(shape)
    assert shape[3] % 2 == 0, "Time must be divisible by 2"
    shape[3] //= 2
    return tuple(shape)


@wraps(field)
def half_field(**kwargs):
    lattice = kwargs.get("lattice", None)
    if lattice is None:
        lattice = lib.lattice
    kwargs["lattice"] = to_half_lattice(lattice)
    return field(**kwargs)
