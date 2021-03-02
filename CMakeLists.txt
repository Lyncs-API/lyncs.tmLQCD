cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(tmLQCD C)

include(ExternalProject)

FILE(GLOB_RECURSE PATCHES "${CMAKE_CURRENT_SOURCE_DIR}/patches/*.patch")
string(REPLACE ";" " " PATCHES "${PATCHES}")

enable_language(Fortran)

find_package(BLAS REQUIRED)
if(NOT BLAS_FOUND)
  message(ERROR "BLAS not found. Install BLAS or load respective modules.")
endif()


set(C_COMPILER "${CMAKE_C_COMPILER}")
set(ENABLE_MPI "no")
find_package(MPI)
if(FALSE AND MPI_FOUND)
  set(C_COMPILER "${MPI_C_COMPILER}")
  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${MPI_C_COMPILER_FLAGS}")
  set(ENABLE_MPI "yes")
endif()


set(ENABLE_OMP "no")
#find_package(OpenMP)
#if(OpenMP_FOUND)
#  set(COMPILER_FLAGS "${COMPILER_FLAGS} ${OpenMP_C_FLAGS}")
#  set(ENABLE_OMP "yes")
#endif()


string (REPLACE ";" " " BLAS_LIBRARIES "${BLAS_LIBRARIES}")

configure_file(config.py.in ${EXTERNAL_INSTALL_LOCATION}/config.py)

ExternalProject_Add(tmLQCD
  GIT_REPOSITORY https://github.com/etmc/tmLQCD
  GIT_TAG c1b0609
  PATCH_COMMAND /bin/sh -c "( git apply ${PATCHES} || git apply ${PATCHES} -R --check ) && ./replace_comm_world.sh && git add --all"
  CONFIGURE_COMMAND /bin/sh -c "autoconf && ./configure \
    CC=${C_COMPILER} \
    F77=${CMAKE_Fortran_COMPILER} \
    CFLAGS=\"-std=gnu99 -O3 -ffast-math -fopenmp ${COMPILER_FLAGS}\" \
    LIBS=\"${BLAS_LIBRARIES}\" \
    --prefix=${EXTERNAL_INSTALL_LOCATION} \
    --enable-mpi=${ENABLE_MPI} \
    --enable-omp=${ENABLE_OMP} \
    --with-limedir=${LIME_PATH}"
  BUILD_COMMAND make -j && make -j install
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

set(TMLQCD_PATH ${EXTERNAL_INSTALL_LOCATION})
set(TMLQCD_INSTALL ON)


