import sys

from lyncs_setuptools import setup, CMakeExtension
from lyncs_clime import __path__ as lime_path

setup(
    "lyncs_tmLQCD",
    exclude=["*.config"],
    ext_modules=[
        CMakeExtension("lyncs_tmLQCD.lib", ".", ["-DLIME_PATH=%s" % lime_path[0],])
    ],
    data_files=[(".", ["config.py.in"])],
    install_requires=["lyncs-cppyy", "lyncs-clime", "numpy",],
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
