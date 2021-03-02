"""
Here we load the library libtmLQCD.so and used headers
"""
# pylint: disable=assigning-non-slot

__all__ = [
    "lib",
    "libBLAS",
    "PATHS",
]

from time import time
from lyncs_clime import lib as libclime
from lyncs_cppyy import Lib, nullptr
from lyncs_cppyy.ll import cast

from . import __path__
from .config import BLAS_LIBRARIES, WITH_MPI


class tmLQCDLib(Lib):
    "Extension of Lib with initialize for tmLQCD"

    __slots__ = [
        "_initialized",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialized = False

    @property
    def initialized(self):
        "Whether the global structure of tmLQCD has been initialized"
        return self._initialized

    def initialize(self, x, y, z, t, comm=None, seed=None):
        "Initializes the global structure of tmLQCD"

        if self.initialized:
            if (x, y, z, t) != (self.LX, self.LY, self.LZ, self.T):
                raise RuntimeError(
                    f"""
                    tmLQCD has been already initialized with
                    (x,y,z,t) = {(self.LX, self.LY, self.LZ, self.T)}
                    and cannot be initialized again.
                    """
                )
            return
        if not x == y == z:
            raise ValueError(
                """
                tmLQCD supports only Lx == Ly == Lz.
                This limitation is due to the usage in the library
                of the variable L. TODO: fix it (good_first_issue).
                """
            )

        self.L = x
        self.LX, self.LY, self.LZ, self.T_global = x, y, z, t

        if WITH_MPI:
            from mpi4py import MPI

            if not comm:
                comm = MPI.COMM_WORLD
            self.g_comm = cast["MPI_Comm"](MPI._handleof(comm))
            self.g_nproc_x, self.g_nproc_y, self.g_nproc_z, self.g_nproc_t = 1, 1, 1, 1
        elif comm:
            raise ValueError(
                "Got a communicator, but tmLQCD has been compiled without MPI."
            )
        else:
            self.g_nproc_x, self.g_nproc_y, self.g_nproc_z, self.g_nproc_t = 1, 1, 1, 1

        self.g_proc_id = 0
        self.g_mu, self.g_kappa = 0, 1
        self.tmlqcd_mpi_init(0, nullptr)
        self.start_ranlux(1, seed or int(time()))
        self.init_geometry_indices(self.VOLUMEPLUSRAND + self.g_dbw2rand)
        self.geometry()
        self.init_gauge_field(
            self.VOLUMEPLUSRAND + self.g_dbw2rand, 0
        )  # 0 for _GAUGE_COPY
        self._initialized = True


PATHS = list(__path__)

headers = [
    "global.h",
    "start.h",
    "mpi_init.h",
    "geometry_eo.h",
    "gamma.h",
    "io/gauge.h",
    "io/params.h",
    "init/init_geometry_indices.h",
    "init/init_gauge_field.h",
    "linalg/convert_even_to_lexic.h",
    "linalg/convert_eo_to_lexic.h",
    "linalg/convert_odd_to_lexic.h",
    "linalg/set_even_to_zero.h",
    "rational/elliptic.h",
    "read_input.h",
    "measure_gauge_action.h",
    "measure_rectangles.h",
    "operator.h",
    "rational/elliptic.h",
    "rational/zolotarev.h",
    # "smearing/stout.h",
]

defined = {}
with open(__path__[0] + "/lib/redefine-syms.txt", "r") as fp:
    defined.update((line.split() for line in fp.readlines()))

libBLAS = Lib(
    header="cblas.h",
    library=BLAS_LIBRARIES.split(),
    c_include=True,
)

libraries = ["libtmLQCD.so", libclime, libBLAS]

if WITH_MPI:
    from lyncs_mpi import lib as libmpi

    libraries.append(libmpi)
    defined["TM_USE_MPI"] = True


lib = tmLQCDLib(
    path=PATHS,
    header=headers,
    library=libraries,
    c_include=True,
    check="measure_plaquette",
    defined=defined,
)
