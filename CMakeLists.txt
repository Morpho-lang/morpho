cmake_minimum_required(VERSION 3.23)
project(morpho-libmorpho)

add_library(morpho SHARED "") 
add_subdirectory(src)

# Include morpho header files across the project 
include_directories("src")
include_directories("src/builtin")
include_directories("src/classes")
include_directories("src/core")
include_directories("src/datastructures")
include_directories("src/debug")
include_directories("src/geometry")
include_directories("src/linalg")
include_directories("src/support")

# Create source groups for IDE
source_group("builtin" REGULAR_EXPRESSION src/builtin/.*\\.[ch])
source_group("classes" REGULAR_EXPRESSION src/classes/.*\\.[ch])
source_group("core" REGULAR_EXPRESSION src/core/.*\\.[ch])
source_group("datastructures" REGULAR_EXPRESSION src/datastructures/.*\\.[ch])
source_group("debug" REGULAR_EXPRESSION src/debug/.*\\.[ch])
source_group("geometry" REGULAR_EXPRESSION src/geometry/.*\\.[ch])
source_group("linalg" REGULAR_EXPRESSION src/linalg/.*\\.[ch])
source_group("support" REGULAR_EXPRESSION src/support/.*\\.[ch])

# Find suitesparse cs.h header file 
find_file(SUITESPARSE_HEADER cs.h 
          HINTS 
            /usr/local/include
            /usr/local/include/suitesparse
            /usr/include/suitesparse)

# Identify folder that cs.h is located in from SUITESPARSE_HEADER and store in SUITESPARSE_INCLUDE
get_filename_component(SUITESPARSE_INCLUDE ${SUITESPARSE_HEADER} DIRECTORY)

# Add morpho headers to MORPHO_INCLUDE
target_include_directories(morpho PUBLIC ${SUITESPARSE_INCLUDE})

# Locate a lapack version
# Currently we prefer LAPACKE 
# TODO: Fix morpho source to select between lapack and lapacke
find_library(LAPACK_LIBRARY
    NAMES lapacke liblapacke lapack liblapack 
)

# Locate cblas
find_library(CBLAS_LIBRARY
    NAMES cblas libcblas blas libblas
)

# Locate suitesparse
find_library(SUITESPARSE_LIBRARY
    NAMES cxsparse libcxsparse
)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(morpho ${SUITESPARSE_LIBRARY} ${CBLAS_LIBRARY} ${LAPACK_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)

# Install the resulting library
install(TARGETS morpho)

# Install morpho header files
install(TARGETS morpho 
        FILE_SET public_headers 
        DESTINATION include/morpho
)
