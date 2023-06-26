cmake_minimum_required(VERSION 3.24)
project(morpho-libmorpho)

add_library(morpho SHARED "") 
add_subdirectory(src)

include_directories("src")
include_directories("src/builtin")
include_directories("src/datastructures")
include_directories("src/geometry")
include_directories("src/utils")
include_directories("src/vm")

include_directories("/usr/local/include")

# Locate lapack
find_library(LAPACK_LIBRARY
    NAMES lapack liblapack
)

# Locate cblas
find_library(CBLAS_LIBRARY
    NAMES cblas libcblas
)

# Locate suitesparse
find_library(SUITESPARSE_LIBRARY
    NAMES cxsparse libcxsparse
)

target_link_libraries(morpho ${SUITESPARSE_LIBRARY} ${CBLAS_LIBRARY} ${LAPACK_LIBRARY} ${CMAKE_DL_LIBS})

# Install the resulting library
install(TARGETS morpho)
