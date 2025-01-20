cmake_minimum_required(VERSION 3.13)
project(morpho-libmorpho)

option(MORPHO_DISABLENANBOXING "Disables NAN Boxing" OFF)

option(MORPHO_GCSTRESSTEST "Stress tests the garbage collector" OFF)

option(MORPHO_NOLINALG "Builds without linear algebra" OFF)
option(MORPHO_NOSPARSE "Builds without sparse matrices" OFF)

set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)

add_library(morpho SHARED "") 
add_subdirectory(src)

# Include morpho header files across the project 
target_include_directories(morpho PUBLIC src 
                                  src/builtin
                                  src/classes
                                  src/core
                                  src/datastructures
                                  src/debug
                                  src/geometry
                                  src/linalg
                                  src/support )

# Option to disable NAN Boxing
if(MORPHO_DISABLENANBOXING)
target_compile_definitions(morpho PUBLIC _NO_NAN_BOXING)
endif() 

# Option to stress test Garbage Collector
if(MORPHO_GCSTRESSTEST)
target_compile_definitions(morpho PUBLIC _DEBUG_STRESSGARBAGECOLLECTOR)
endif() 

# Set help directory
if(MORPHO_HELP_BASEDIR)
target_compile_definitions(morpho PUBLIC MORPHO_HELP_BASEDIR=\"${MORPHO_HELP_BASEDIR}\")
endif() 

if(MORPHO_MODULE_BASEDIR)
target_compile_definitions(morpho PUBLIC MORPHO_MODULE_BASEDIR=\"${MORPHO_MODULE_BASEDIR}\")
endif() 

# Create source groups for IDE
source_group("builtin" REGULAR_EXPRESSION src/builtin/.*\\.[ch])
source_group("classes" REGULAR_EXPRESSION src/classes/.*\\.[ch])
source_group("core" REGULAR_EXPRESSION src/core/.*\\.[ch])
source_group("datastructures" REGULAR_EXPRESSION src/datastructures/.*\\.[ch])
source_group("debug" REGULAR_EXPRESSION src/debug/.*\\.[ch])
source_group("geometry" REGULAR_EXPRESSION src/geometry/.*\\.[ch])
source_group("linalg" REGULAR_EXPRESSION src/linalg/.*\\.[ch])
source_group("support" REGULAR_EXPRESSION src/support/.*\\.[ch])

if (!MORPHO_NOSPARSE)
# Find suitesparse cs.h header file 
find_file(SUITESPARSE_HEADER cs.h 
          HINTS 
            /home/linuxbrew/.linuxbrew/include/suitesparse 
            /usr/local/include
            /usr/local/include/suitesparse
            /opt/homebrew/include/suitesparse
            /usr/include/suitesparse)

# Identify folder that cs.h is located in from SUITESPARSE_HEADER and store in SUITESPARSE_INCLUDE
get_filename_component(SUITESPARSE_INCLUDE ${SUITESPARSE_HEADER} DIRECTORY)

# Add suitesparse headers to include folders
target_include_directories(morpho PUBLIC ${SUITESPARSE_INCLUDE})

# Locate suitesparse
find_library(SUITESPARSE_LIBRARY
    NAMES cxsparse libcxsparse
)
endif()

if (!MORPHO_NOLINALG)
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
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(morpho ${SUITESPARSE_LIBRARY} ${CBLAS_LIBRARY} ${LAPACK_LIBRARY} ${CMAKE_DL_LIBS} Threads::Threads)

# Install the resulting library
install(TARGETS morpho)

# Install morpho header files
# The below works in 3.23, which is too recent for Ubuntu
# install(TARGETS morpho 
#        FILE_SET public_headers 
#        DESTINATION include/morpho
#)

# So we'll use a hacky way for now
file(GLOB_RECURSE MORPHO_HEADER_FILES "src/*.h")
install(FILES ${MORPHO_HEADER_FILES} DESTINATION include/morpho)

# Similarly install morpho modules
file(GLOB_RECURSE MORPHO_MODULES_FILES "modules/*.morpho")
install(FILES ${MORPHO_MODULES_FILES} DESTINATION share/morpho/modules)

# Similarly install morpho help files
file(GLOB_RECURSE MORPHO_HELP_FILES "help/*.md")
install(FILES ${MORPHO_HELP_FILES} DESTINATION share/morpho/help)
