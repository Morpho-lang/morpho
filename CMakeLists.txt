cmake_minimum_required(VERSION 3.20)

project(morpho LANGUAGES C VERSION 0.5.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(BLAS REQUIRED)
find_package(SuiteSparse REQUIRED COMPONENTS CXSparse)
set(MORPHO_LIBRARIES m BLAS::BLAS SuiteSparse::SuiteSparse)

if(APPLE)
    # this should find the Accelerate framework on Apple
    find_package(LAPACK REQUIRED)
    list(APPEND MORPHO_LIBRARIES LAPACK::LAPACK)
else()
    find_package(LAPACKE REQUIRED)
    list(APPEND MORPHO_LIBRARIES LAPACKE::LAPACKE)
endif()

find_package(SuiteSparse REQUIRED COMPONENTS CXSparse)

include(GNUInstallDirs)

set(MORPHO_MODULE_DIR ${CMAKE_INSTALL_FULL_DATADIR}/morpho/modules)
set(MORPHO_HELP_DIR ${CMAKE_INSTALL_FULL_DATADIR}/morpho/help)

add_subdirectory(deps)
add_subdirectory(morpho5)
add_subdirectory(morphoview)

install(TARGETS morpho morphoview RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/morpho5/modules
        DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/morpho5/docs/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho/help
        FILES_MATCHING PATTERN "*.md")
