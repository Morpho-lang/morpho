cmake_minimum_required(VERSION 3.13)
project(morpho-libmorpho)

add_library(morpho SHARED "") 
add_subdirectory(src)

# Include morpho header files across the project 
include_directories("src")
include_directories("src/builtin")
include_directories("src/datastructures")
include_directories("src/geometry")
include_directories("src/utils")
include_directories("src/vm")

# Find suitesparse cs.h header file 
find_file(SUITESPARSE_HEADER cs.h)

# Identify folder that cs.h is located in from SUITESPARSE_HEADER and store in SUITESPARSE_INCLUDE
get_filename_component(SUITESPARSE_INCLUDE ${SUITESPARSE_HEADER} DIRECTORY)

# Add morpho headers to MORPHO_INCLUDE
target_include_directories(morpho PUBLIC ${SUITESPARSE_INCLUDE})

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
