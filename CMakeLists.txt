cmake_minimum_required(VERSION 3.20)

project(morpho LANGUAGES C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(SuiteSparse REQUIRED COMPONENTS CXSparse)
set(MORPHO_LIBRARIES m SuiteSparse::SuiteSparse)

if(APPLE)
    list(APPEND MORPHO_LIBRARIES "-framework Accelerate")
else()
    find_package(BLAS REQUIRED)
    find_package(LAPACKE REQUIRED)

    list(APPEND MORPHO_LIBRARIES BLAS::BLAS LAPACKE::LAPACKE)
endif()

include(GNUInstallDirs)

set(MORPHO_DATA_DIR ${CMAKE_INSTALL_PREFIX}/share/morpho)
set(MORPHO_MODULE_DIR ${MORPHO_DATA_DIR}/modules)
set(MORPHO_HELP_DIR ${MORPHO_DATA_DIR}/help)

add_subdirectory(deps)
add_subdirectory(morpho5)
add_subdirectory(morphoview)

install(TARGETS morpho morphoview RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/morpho5/modules DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/morpho5/docs/ DESTINATION ${CMAKE_INSTALL_DATADIR}/morpho/help FILES_MATCHING PATTERN "*.md")
