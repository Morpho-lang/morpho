#-------------------------------------------------------------------------------
# Morpho/CMakeLists.txt
#-------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.13)

# Default to Release when no build type is set.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "CMAKE_BUILD_TYPE unset; defaulting to Release")
    set(CMAKE_BUILD_TYPE Release CACHE STRING
        "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

project(morpho-libmorpho LANGUAGES C)

include(GNUInstallDirs)

#-------------------------------------------------------------------------------
# Platform dependent options
#-------------------------------------------------------------------------------

add_library(morpho SHARED "")
set_target_properties(morpho PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)
add_library(morpho::morpho ALIAS morpho)

# Detect platform-native shared library extension and set MORPHO_DYLIBEXTENSION to it
if(NOT MORPHO_DYLIBEXTENSION)
    string(REGEX REPLACE "^\\." "" MORPHO_DYLIBEXTENSION "${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
target_compile_definitions(morpho PRIVATE MORPHO_DYLIBEXTENSION=\"${MORPHO_DYLIBEXTENSION}\")

#-------------------------------------------------------------------------------
# Release LTO
#-------------------------------------------------------------------------------

include(CheckIPOSupported)
check_ipo_supported(RESULT MORPHO_IPO_SUPPORTED OUTPUT MORPHO_IPO_OUTPUT LANGUAGES C)
if(MORPHO_IPO_SUPPORTED)
    set_property(TARGET morpho PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    message(STATUS "IPO/LTO enabled for Release")
else()
    message(STATUS "IPO/LTO not enabled (${MORPHO_IPO_OUTPUT})")
endif()
 
#-------------------------------------------------------------------------------
# Options
#-------------------------------------------------------------------------------

option(MORPHO_DISABLENANBOXING "Disables NAN Boxing" OFF)
option(MORPHO_GCSTRESSTEST "Stress tests the garbage collector" OFF)
option(MORPHO_BUILD_LINALG "Builds with linear algebra" ON)
option(MORPHO_BUILD_SPARSE "Builds with sparse matrix support" ON)
option(MORPHO_BUILD_GEOMETRY "Builds with geometry library" ON)
option(MORPHO_BUILD_PROFILER "Builds with profiler" ON)
set(MORPHO_OPENBLAS_ROOT "" CACHE PATH
    "OpenBLAS installation prefix")
set(MORPHO_SUITESPARSE_ROOT "" CACHE PATH
    "SuiteSparse installation prefix")

#-------------------------------------------------------------------------------
# Process options
#-------------------------------------------------------------------------------

# Option to disable NAN Boxing
if(MORPHO_DISABLENANBOXING)
target_compile_definitions(morpho PUBLIC _NO_NAN_BOXING)
endif() 

# Option to stress test Garbage Collector
if(MORPHO_GCSTRESSTEST)
target_compile_definitions(morpho PRIVATE _DEBUG_STRESSGARBAGECOLLECTOR)
endif()

# Optional overrides. Otherwise build.h supplies its defaults.
if(MORPHO_HELP_BASEDIR)
    target_compile_definitions(morpho PRIVATE
        MORPHO_HELP_BASEDIR=\"${MORPHO_HELP_BASEDIR}\")
endif()

if(MORPHO_MODULE_BASEDIR)
    target_compile_definitions(morpho PRIVATE
        MORPHO_MODULE_BASEDIR=\"${MORPHO_MODULE_BASEDIR}\")
endif() 

# Include geometry as part of the build
if(MORPHO_BUILD_GEOMETRY)
target_compile_definitions(morpho PUBLIC MORPHO_INCLUDE_GEOMETRY)
endif()

# Build with profiler
if(MORPHO_BUILD_PROFILER)
target_compile_definitions(morpho PUBLIC _DEBUG_PROFILER)
endif()

#-------------------------------------------------------------------------------
# BLAS and LAPACK
#-------------------------------------------------------------------------------

if (MORPHO_BUILD_LINALG)

message(STATUS "Searching for BLAS and LAPACK")
# linalg_includes.h uses Accelerate on Apple and LAPACKE elsewhere.
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if(NOT ACCELERATE_FRAMEWORK)
        message(FATAL_ERROR
            "Morpho linear algebra requires the Accelerate framework.")
    endif()
    message(STATUS "Found Accelerate at ${ACCELERATE_FRAMEWORK}")
    target_link_libraries(morpho PRIVATE ${ACCELERATE_FRAMEWORK})
else()
    # Explicit root, otherwise the Windows OpenBLAS staging directory.
    set(_MORPHO_OPENBLAS_HINTS "")
    if(MORPHO_OPENBLAS_ROOT)
        list(APPEND _MORPHO_OPENBLAS_HINTS "${MORPHO_OPENBLAS_ROOT}")
    elseif(WIN32 AND EXISTS "C:/openblas-staging")
        list(APPEND _MORPHO_OPENBLAS_HINTS "C:/openblas-staging")
    endif()

    find_library(LAPACK_LIBRARY NAMES lapacke liblapacke openblas libopenblas
        HINTS ${_MORPHO_OPENBLAS_HINTS}
        PATH_SUFFIXES lib)
    find_library(CBLAS_LIBRARY NAMES cblas libcblas openblas libopenblas blas libblas
        HINTS ${_MORPHO_OPENBLAS_HINTS}
        PATH_SUFFIXES lib)
    find_path(CBLAS_INCLUDE NAMES cblas.h
        HINTS ${_MORPHO_OPENBLAS_HINTS}
        PATH_SUFFIXES include include/openblas openblas)
    find_path(LAPACKE_INCLUDE NAMES lapacke.h
        HINTS ${_MORPHO_OPENBLAS_HINTS} "${CBLAS_INCLUDE}"
        PATH_SUFFIXES include include/openblas openblas)

    if(NOT CBLAS_LIBRARY OR NOT LAPACK_LIBRARY OR NOT CBLAS_INCLUDE OR NOT LAPACKE_INCLUDE)
        message(FATAL_ERROR
            "Morpho linear algebra requires CBLAS, LAPACKE, cblas.h, and lapacke.h.")
    endif()

    message(STATUS "Found LAPACK at ${LAPACK_LIBRARY}")
    message(STATUS "Found BLAS at ${CBLAS_LIBRARY}")
    message(STATUS "Found cblas.h at ${CBLAS_INCLUDE}")
    message(STATUS "Found lapacke.h at ${LAPACKE_INCLUDE}")

    target_include_directories(morpho PRIVATE ${CBLAS_INCLUDE})
    if(NOT LAPACKE_INCLUDE STREQUAL CBLAS_INCLUDE)
        target_include_directories(morpho PRIVATE ${LAPACKE_INCLUDE})
    endif()
    target_link_libraries(morpho PRIVATE ${CBLAS_LIBRARY} ${LAPACK_LIBRARY})
endif()

target_compile_definitions(morpho PUBLIC MORPHO_INCLUDE_LINALG)

endif()

#-------------------------------------------------------------------------------
# SuiteSparse
#-------------------------------------------------------------------------------

if (MORPHO_BUILD_SPARSE)
message(STATUS "Searching for Suitesparse")
# Explicit root, otherwise the Windows SuiteSparse staging directory.
set(_MORPHO_SUITESPARSE_HINTS "")
if(MORPHO_SUITESPARSE_ROOT)
    list(APPEND _MORPHO_SUITESPARSE_HINTS "${MORPHO_SUITESPARSE_ROOT}")
elseif(WIN32 AND EXISTS "C:/suitesparse-staging")
    list(APPEND _MORPHO_SUITESPARSE_HINTS "C:/suitesparse-staging")
endif()

find_library(SUITESPARSE_LIBRARY NAMES cxsparse libcxsparse
    HINTS ${_MORPHO_SUITESPARSE_HINTS}
    PATH_SUFFIXES lib)
find_path(SUITESPARSE_INCLUDE NAMES cs.h
    HINTS ${_MORPHO_SUITESPARSE_HINTS}
    PATH_SUFFIXES include/suitesparse suitesparse)
if(NOT SUITESPARSE_LIBRARY OR NOT SUITESPARSE_INCLUDE)
    message(FATAL_ERROR
        "Morpho sparse support requires CXSparse (cxsparse and cs.h).")
endif()
message(STATUS "Found suitesparse at ${SUITESPARSE_LIBRARY}")
message(STATUS "Found cs.h at ${SUITESPARSE_INCLUDE}")

target_include_directories(morpho PRIVATE ${SUITESPARSE_INCLUDE})
target_compile_definitions(morpho PUBLIC MORPHO_INCLUDE_SPARSE)
target_link_libraries(morpho PRIVATE ${SUITESPARSE_LIBRARY})
endif()

#-------------------------------------------------------------------------------
# Threads
#-------------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

#-------------------------------------------------------------------------------
# Add morpho sources
#-------------------------------------------------------------------------------

add_subdirectory(src)

# Include morpho header files across the project
target_include_directories(morpho PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/builtin>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/classes>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/datastructures>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/debug>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geometry>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/functionals>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/linalg>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/support>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> # enable use <morpho/___.h>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/morpho>
)

# Create source groups for IDE
source_group("builtin" REGULAR_EXPRESSION src/builtin/.*\\.[ch])
source_group("classes" REGULAR_EXPRESSION src/classes/.*\\.[ch])
source_group("core" REGULAR_EXPRESSION src/core/.*\\.[ch])
source_group("datastructures" REGULAR_EXPRESSION src/datastructures/.*\\.[ch])
source_group("debug" REGULAR_EXPRESSION src/debug/.*\\.[ch])
source_group("geometry" REGULAR_EXPRESSION src/geometry/.*\\.[ch])
source_group("geometry/functionals" REGULAR_EXPRESSION src/geometry/functionals/.*\\.[ch])
source_group("linalg" REGULAR_EXPRESSION src/linalg/.*\\.[ch])
source_group("support" REGULAR_EXPRESSION src/support/.*\\.[ch])

target_link_libraries(morpho PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------

# Install the resulting library and its CMake package config.
set(MORPHO_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/morpho"
)

install(TARGETS morpho
    EXPORT morphoTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT morphoTargets
    FILE morphoTargets.cmake
    NAMESPACE morpho::
    DESTINATION "${MORPHO_CMAKE_INSTALL_DIR}"
)

install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/morphoConfig.cmake"
    DESTINATION "${MORPHO_CMAKE_INSTALL_DIR}"
)

# Install public headers. linalg_includes.h stays private.
file(GLOB_RECURSE MORPHO_HEADER_FILES "src/*.h")
list(FILTER MORPHO_HEADER_FILES EXCLUDE REGEX "/linalg_includes\\.h$")
install(FILES ${MORPHO_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/morpho
)

# Similarly install morpho modules
file(GLOB_RECURSE MORPHO_MODULES_FILES "modules/*.morpho")
install(FILES ${MORPHO_MODULES_FILES}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho/modules
)

# Similarly install morpho help files
file(GLOB_RECURSE MORPHO_HELP_FILES "help/*.md")
install(FILES ${MORPHO_HELP_FILES}
    DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho/help
)
