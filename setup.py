import sys

from lyncs_setuptools import setup, CMakeExtension, find_package
from lyncs_clime import __path__ as lime_path

install_requires = [
    "dataclasses",
    "lyncs-setuptools",
    "lyncs-cppyy",
    "lyncs-clime",
    "numpy",
]

findMPI = find_package("MPI")
if findMPI["found"]:
    install_requires.append("lyncs_mpi")

setup(
    "lyncs_tmLQCD",
    exclude=["*.config"],
    data_files=[(".", ["config.py.in"])],
    ext_modules=[
        CMakeExtension(
            "lyncs_tmLQCD.lib",
            ".",
            [
                "-DLIME_PATH=%s" % lime_path[0],
            ],
        )
    ],
    install_requires=install_requires,
    extras_require={
        "test": ["pytest", "pytest-cov", "lyncs-setuptools[pylint]", "scipy"],
    },
    keywords=[
        "Lyncs",
        "tmLQCD",
        "Lattice QCD",
        "Wilson",
        "Twisted-mass",
        "Clover",
        "Fermions",
        "HMC",
        "Actions",
        "ETMC",
    ],
)
