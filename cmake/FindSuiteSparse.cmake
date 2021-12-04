# FindSuiteSparse.cmake
#
# Attempts to find the SuiteSparse library. If found, sets the following
# variables:
#   SuiteSparse_FOUND        : set if the library was found
#   SuiteSparse_INCLUDE_DIRS : list of include directories
#   SuiteSparse_LIBRARIES    : SuiteSparse libraries to link to

# TODO: Make this more robust in the future. This doesn't check the version
#       or for any dependencies, it just finds headers and library files. It
#       doesn't look at optional component either!

include(FindPackageHandleStandardArgs)

set(SUITESPARSE_COMPONENTS
    AMD
    BTF
    CAMD
    CCOLAMD
    CHOLAMD
    CSparse
    CXSparse
    GPUQREngine
    GraphBLAS
    KLU
    LDL
    SPQR
    UMFPACK
)

# representative header file for each component
set(AMD_HEADER "amd.h")
set(BTF_HEADER "btf.h")
set(CAMD_HEADER "camd.h")
set(CCOLAMD_HEADER "ccolamd.h")
set(CHOLAMD_HEADER "cholamd.h")
set(CSparse_HEADER "cs.h")
set(CXSparse_HEADER "cs.h")
set(GPUQREngine_HEADER "GPUQREngine.hpp")
set(GraphBLAS_HEADER "GraphBLAS.h")
set(KLU_HEADER "klu.h")
set(LDL_HEADER "ldl.h")
set(SPQR_HEADER "spqr.hpp")
set(UMFPACK_HEADER "umfpack.h")

# representative library file for each component
set(AMD_LIB "amd")
set(BTF_LIB "btf")
set(CAMD_LIB "camd")
set(CCOLAMD_LIB "ccolamd")
set(CHOLAMD_LIB "cholamd")
set(CSparse_LIB "csparse")
set(CXSparse_LIB "cxsparse")
set(GPUQREngine_LIB "GPUQREngine")
set(GraphBLAS_LIB "GraphBLAS")
set(KLU_LIB "klu")
set(LDL_LIB "ldl")
set(SPQR_LIB "spqr")
set(UMFPACK_LIB "umfpack")

# core config library for each SuiteSparse component
find_library(SUITESPARSE_CONFIG_LIBRARY "suitesparseconfig" REQUIRED)
find_path(SUITESPARSE_CONFIG_INCLUDE_DIR "SuiteSparse_config.h" REQUIRED
    PATH_SUFFIXES "suitesparse"
)

mark_as_advanced(FORCE
    SUITESPARSE_CONFIG_LIBRARY
    SUITESPARSE_CONFIG_INCLUDE_DIR
)

foreach(_component ${SUITESPARSE_COMPONENTS})
    find_library(SuiteSparse_${_component}_LIBRARY "${${_component}_LIB}")
    find_path(SuiteSparse_${_component}_INCLUDE_DIR "${${_component}_HEADER}"
        PATH_SUFFIXES "suitesparse"
    )

    if(SuiteSparse_${_component}_LIBRARY AND SuiteSparse_${_component}_INCLUDE_DIR)
        set(SuiteSparse_${_component}_FOUND TRUE)
    else()
        set(SuiteSparse_${_component}_FOUND FALSE)
    endif()

    mark_as_advanced(FORCE
        SuiteSparse_${_component}_LIBRARY
        SuiteSparse_${_component}_INCLUDE_DIR
    )
endforeach()

find_package_handle_standard_args(SuiteSparse
    REQUIRED_VARS
        SUITESPARSE_CONFIG_LIBRARY
        SUITESPARSE_CONFIG_INCLUDE_DIR
    HANDLE_COMPONENTS
)

if(NOT TARGET SuiteSparse::SuiteSparse)
    add_library(SuiteSparse::SuiteSparse INTERFACE IMPORTED)
    target_include_directories(SuiteSparse::SuiteSparse INTERFACE ${SUITESPARSE_CONFIG_INCLUDE_DIR})
    target_link_libraries(SuiteSparse::SuiteSparse INTERFACE ${SUITESPARSE_CONFIG_LIBRARY})
endif()

set(SuiteSparse_LIBRARIES "")
set(SuiteSparse_INCLUDE_DIRS "")
foreach(_component ${SuiteSparse_FIND_COMPONENTS})
    set(_target SuiteSparse::${_component})
    
    if(NOT TARGET ${_target})
        add_library(${_target} UNKNOWN IMPORTED)
        set_target_properties(${_target} PROPERTIES
            IMPORTED_LOCATION ${SuiteSparse_${_component}_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${SuiteSparse_${_component}_INCLUDE_DIR}
            INTERFACE_LINK_LIBRARIES ${SUITESPARSE_CONFIG_LIBRARY}
        )
    endif()

    target_link_libraries(SuiteSparse::SuiteSparse INTERFACE ${_target})

    list(APPEND SuiteSparse_LIBRARIES ${SuiteSparse_${_component}_LIBRARY})
    list(APPEND SuiteSparse_INCLUDE_DIRS ${SuiteSparse_${_component}_INCLUDE_DIR})
endforeach()
