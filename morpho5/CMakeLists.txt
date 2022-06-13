# TODO: This is not considered good practice!
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.h" "*.c")
list(REMOVE_ITEM sources "main.c")

if(NOT MSVC)
    set(MORPHO_WARNINGS
        -Wall
        -Wextra
        # -Wpedantic
        -Wno-sign-compare
        -Wno-unused-parameter
    )
else()
    set(MORPHO_WARNINGS \Wx \W4)
endif()

add_library(morpho-lib STATIC ${sources})

target_include_directories(morpho-lib PUBLIC
    .
    builtin
    datastructures
    geometry
    interface
    utils
    vm
)

target_compile_definitions(morpho-lib PUBLIC
    MORPHO_HELPDIRECTORY="${MORPHO_HELP_DIR}"
    MORPHO_MODULEDIRECTORY="${MORPHO_MODULE_DIR}"
)

target_compile_options(morpho-lib PUBLIC ${MORPHO_WARNINGS})
target_link_libraries(morpho-lib PUBLIC ${MORPHO_LIBRARIES})

add_executable(morpho main.c)
target_link_libraries(morpho PRIVATE morpho-lib)
