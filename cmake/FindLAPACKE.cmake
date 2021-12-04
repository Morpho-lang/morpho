# FindLAPACKE.cmake
#
# Attempts to find the LAPACKE standard C interface header. If found, sets the
# following variables:
#   LAPACKE_FOUND        : set if the library was found
#   LAPACKE_INCLUDE_DIRS : list of include directories
#   LAPACKE_LIBRARIES    : SuiteSparse libraries to link to

# TODO: I think this will need to be adjusted to find, for example, the MKL
#       version of the lapacke headers.

include(FindPackageHandleStandardArgs)

find_path(LAPACKE_INCLUDE_DIRS "lapacke.h" REQUIRED PATH_SUFFIXES "lapacke")
find_library(LAPACKE_LIBRARIES "lapacke" REQUIRED PATH_SUFFIXES "lapacke")

mark_as_advanced(FORCE LAPACKE_INCLUDE_DIRS)
mark_as_advanced(FORCE LAPACKE_LIBRARIES)

find_package_handle_standard_args(LAPACKE
    REQUIRED_VARS
        LAPACKE_LIBRARIES
        LAPACKE_INCLUDE_DIRS
)

add_library(LAPACKE::LAPACKE INTERFACE IMPORTED)
target_include_directories(LAPACKE::LAPACKE INTERFACE ${LAPACKE_INCLUDE_DIRS})
target_link_libraries(LAPACKE::LAPACKE INTERFACE ${LAPACKE_LIBRARIES})
