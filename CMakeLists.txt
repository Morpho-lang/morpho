cmake_minimum_required(VERSION 3.13)

project(morpho-newlinalg)

# Build the library as a plugin
add_library(newlinalg MODULE "") 

# Suppress 'lib' prefix
set_target_properties(newlinalg PROPERTIES PREFIX "")

# Add sources
add_subdirectory(src)

#-------------------------------------------------------------------------------
# Morpho library
#-------------------------------------------------------------------------------

# Locate the morpho.h header file and store in MORPHO_HEADER
find_file(MORPHO_HEADER
          morpho.h 
          HINTS 
          /usr/local/opt/morpho
          /opt/homebrew/opt/morpho
          /usr/local/include/morpho
          "C:\\Program Files\\Morpho\\include\\"
          )

# Identify folder that morpho.h is located in from MORPHO_HEADER and store in MORPHO_INCLUDE
get_filename_component(MORPHO_INCLUDE ${MORPHO_HEADER} DIRECTORY)

# Add morpho headers to MORPHO_INCLUDE
target_include_directories(newlinalg PUBLIC ${MORPHO_INCLUDE})

# Add general header search paths
target_include_directories(newlinalg PUBLIC /usr/local/include /opt/homebrew/include)

# Add morpho headers in subfolders to MORPHO_INCLUDE
file(GLOB morpho_subdirectories LIST_DIRECTORIES true ${MORPHO_INCLUDE}/*)
foreach(dir ${morpho_subdirectories})
    IF(IS_DIRECTORY ${dir})
        target_include_directories(zeromq PUBLIC ${dir})
    ELSE()
        CONTINUE()
    ENDIF()
endforeach()

# Locate libmorpho
find_library(MORPHO_LIBRARY
    NAMES morpho libmorpho
)

target_link_libraries(newlinalg ${MORPHO_LIBRARY})

#-------------------------------------------------------------------------------
# BLAS and LAPACK
#-------------------------------------------------------------------------------

message(STATUS "Searching for BLAS and LAPACK")
# Locate a lapack version
# Currently we prefer LAPACKE
# TODO: Fix morpho source to select between lapack and lapacke
find_library(LAPACK_LIBRARY
    NAMES lapacke liblapacke lapack liblapack libopenblas
    HINTS
        "C:\\Program Files\\Morpho\\lib\\"
)
if (LAPACK_LIBRARY)
message(STATUS "Found LAPACK at ${LAPACK_LIBRARY}")
endif()

# Locate cblas
find_library(CBLAS_LIBRARY
    NAMES cblas libcblas blas libblas openblas libopenblas
    HINTS
        "C:\\Program Files\\Morpho\\lib\\"
)
if (CBLAS_LIBRARY)
message(STATUS "Found blas at ${CBLAS_LIBRARY}")
endif()

# Find cblas.h header file
find_path(CBLAS_INCLUDE cblas.h
          HINTS
            "C:\\Program Files\\Morpho\\include\\lapack")

# Add cblas headers to include folders
target_include_directories(newlinalg PUBLIC ${CBLAS_INCLUDE})

message(STATUS "Found cblas headers at ${CBLAS_INCLUDE}")

target_link_libraries(newlinalg ${CBLAS_LIBRARY})
target_link_libraries(newlinalg ${LAPACK_LIBRARY})

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------

set(CMAKE_INSTALL_PREFIX ..)

# Install the resulting binary
install(TARGETS newlinalg LIBRARY DESTINATION lib/)
