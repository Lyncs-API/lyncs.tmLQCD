"""
Functions for gauge field
"""

__all__ = [
    "Gauge",
    "get_g_gauge_field",
]

from numpy import prod, frombuffer
from lyncs_cppyy.ll import to_pointer, addressof, free
from lyncs_utils import static_property
from .base import Field
from .lib import lib


class Gauge(Field):
    "Interface for gauge fields"

    @static_property
    def field_shape():
        "Shape of the field"
        return (4, 3, 3)

    @property
    def su3_field(self):
        "su3** view of the field"
        return to_pointer(self.ptr, "su3 **")

    def volume_plaquette(self, coeff=0):
        """
        Returns the sum over the plaquette.
        The coefficient is used to weight differently the spatial and temporal plaquette,
        having P(c) = (1+c) P_time + (1-c) P_space
        """
        if coeff == 0:
            return lib.measure_plaquette(self.su3_field)
        return lib.measure_gauge_action(self.su3_field, coeff)

    def plaquette(self):
        "Returns the averaged plaquette"
        return self.volume_plaquette() / self.volume / 6

    def temporal_plaquette(self):
        "Returns the averaged temporal plaquette"
        return self.volume_plaquette(1) / self.volume / 6

    def spatial_plaquette(self):
        "Returns the averaged spatial plaquette"
        return self.volume_plaquette(-1) / self.volume / 6

    def volume_rectangles(self):
        "Returns the sum over the rectangles"
        return lib.measure_rectangles(self.su3_field)

    def rectangles(self):
        "Returns the averaged rectangles"
        return self.volume_rectangles() / self.volume / 12

    def gauge_action(self, plaq_coeff=0, rect_coeff=0):
        """
        Returns the gauge action.

        The coefficients are use as follows

            (1-8*c1) ((1+c0) P_time + (1-c0) P_space) + c1*R

        where P is the sum over plaquette, R over the rectangles,
        c0 is the plaq_coeff (see volume_plaquette) and c1 the rect_coeff
        """
        return (1 - 8 * rect_coeff) * self.volume_plaquette(plaq_coeff) + (
            (rect_coeff * self.volume_rectangles()) if rect_coeff != 0 else 0
        )

    def symanzik_gauge_action(self, plaq_coeff=0):
        "Returns the tree-level Symanzik improved gauge action"
        return self.gauge_action(plaq_coeff, -1 / 12)

    def iwasaki_gauge_action(self, plaq_coeff=0):
        "Returns the Iwasaki gauge action"
        return self.gauge_action(plaq_coeff, -0.331)

    def unity(self):
        "Creates a unity field"
        self[:] = 0
        self.reshape(-1, 9)[:, (0, 4, 8)] = 1

    def random(self, repro=False):
        "Creates a random field"
        lib.random_gauge_field(repro, self.su3_field)

    def copy_to_global(self):
        "Copies the field to the global gauge field"
        get_g_gauge_field()[:] = self[:]

    def copy_from_global(self):
        "Copies the global gauge field to the local field"
        self[:] = get_g_gauge_field()[:]

    def write(self, filename, number=0):
        "Writes to file in lime format"
        self.copy_to_global()
        xlfInfo = lib.construct_paramsXlfInfo(self.plaquette(), number)
        lib.write_gauge_field(filename, 64, xlfInfo)
        free(xlfInfo)

    def read(self, filename):
        "Reads from file in lime format"
        lib.gauge_precision_read_flag = 64
        lib.read_gauge_field(filename, self.su3_field)

    # def stout_smearing(self, rho=0.1, niters=1):
    #     """
    #     Returns a new gauge field with stout smearing using
    #     the coefficient rho and the given number of iterations
    #     """
    #     out = Gauge(empty_like(self))
    #     params = lib.stout_parameters(rho,niters)
    #     lib.stout_smear(out.su3_field, params, self.su3_field)
    #     return out


def get_g_gauge_field():
    "Returns the global gauge field in usage by tmLQCD"
    assert lib.initialized
    shape = (lib.LX, lib.LY, lib.LZ, lib.T, 4, 3, 3)
    ptr = to_pointer(addressof(lib.g_gauge_field))
    ptr.reshape((int(prod(shape)) * 2,))
    return Gauge(frombuffer(ptr, dtype="complex", count=prod(shape)).reshape(shape))


def get_g_iup():
    "Returns the neighbouring indexes defined by tmLQCD"
    assert lib.initialized
    shape = (lib.LX, lib.LY, lib.LZ, lib.T, 4)
    ptr = to_pointer(addressof(lib.g_iup))
    ptr.reshape((int(prod(shape)),))
    return frombuffer(ptr, dtype="int32", count=prod(shape)).reshape(shape)
