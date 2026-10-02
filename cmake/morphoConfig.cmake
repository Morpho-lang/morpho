#-------------------------------------------------------------------------------
# Morpho/cmake/morphoConfig.cmake
#
# Package config imported by find_package(morpho). Loads the exported
# morpho::morpho target and, when needed, locates cblas.h and lapacke.h.
#-------------------------------------------------------------------------------

include("${CMAKE_CURRENT_LIST_DIR}/morphoTargets.cmake")

get_target_property(_morpho_defs morpho::morpho INTERFACE_COMPILE_DEFINITIONS)
if(_morpho_defs MATCHES "(^|;)MORPHO_INCLUDE_LINALG($|;)" AND NOT APPLE)
    get_filename_component(_morpho_prefix "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)
    if(NOT EXISTS "${_morpho_prefix}/include/openblas/cblas.h")
        find_path(MORPHO_CBLAS_INCLUDE NAMES cblas.h PATH_SUFFIXES openblas)
        find_path(MORPHO_LAPACKE_INCLUDE NAMES lapacke.h
            HINTS "${MORPHO_CBLAS_INCLUDE}"
            PATH_SUFFIXES openblas
        )
        if(NOT MORPHO_CBLAS_INCLUDE OR NOT MORPHO_LAPACKE_INCLUDE)
            message(FATAL_ERROR
                "Morpho was built with linear algebra, but cblas.h or lapacke.h was not found.")
        endif()
        target_include_directories(morpho::morpho INTERFACE "${MORPHO_CBLAS_INCLUDE}")
        if(NOT MORPHO_LAPACKE_INCLUDE STREQUAL MORPHO_CBLAS_INCLUDE)
            target_include_directories(morpho::morpho INTERFACE "${MORPHO_LAPACKE_INCLUDE}")
        endif()
    endif()
endif()
unset(_morpho_defs)
unset(_morpho_prefix)
