# A Python interface to tmLQCD for Lyncs

[![python](https://img.shields.io/pypi/pyversions/lyncs_tmLQCD.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_tmLQCD/)
[![pypi](https://img.shields.io/pypi/v/lyncs_tmLQCD.svg?logo=python&logoColor=white)](https://pypi.org/project/lyncs_tmLQCD/)
[![license](https://img.shields.io/github/license/Lyncs-API/lyncs.tmLQCD?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.tmLQCD/blob/master/LICENSE)
[![build & test](https://img.shields.io/github/workflow/status/Lyncs-API/lyncs.tmLQCD/build%20&%20test?logo=github&logoColor=white)](https://github.com/Lyncs-API/lyncs.tmLQCD/actions)
[![codecov](https://img.shields.io/codecov/c/github/Lyncs-API/lyncs.tmLQCD?logo=codecov&logoColor=white)](https://codecov.io/gh/Lyncs-API/lyncs.tmLQCD)
[![pylint](https://img.shields.io/badge/pylint%20score-6.6%2F10-orange?logo=python&logoColor=white)](http://pylint.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg?logo=codefactor&logoColor=white)](https://github.com/ambv/black)


[tmLQCD] is the simulation library used by the Extended Twisted Mass Collaboration (ETMC).
tmLQCD is a freely available software suite providing a set of tools to be used in lattice QCD
simulations. This is mainly a (P/R)HMC implementation for Wilson and Wilson twisted mass fermions
and inverter for different versions of the Dirac operator.

The code is fully parallelised and ships with optimisations for various modern architectures,
such as commodity PC clusters and the Blue Gene family.

[tmLQCD]: https://github.com/etmc/tmLQCD


## Installation

The package can be installed via `pip`:

```
pip install [--user] lyncs_tmLQCD
```

### External dependencies

For compiling tmLQCD, a fortran compiler, flex, openblas and lapack are required.

These can be installed via `apt`:

```
apt install -y flex libopenblas-dev liblapack-dev gfortran
```

OR using `conda`:

```
conda install -c anaconda openblas
conda install -c conda-forge flex lapack fortran-compiler
```

## Documentation

