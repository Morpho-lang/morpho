file(GLOB_RECURSE sources CONFIGURE_DEPENDS "*.h" "*.c")

find_package(glfw3 REQUIRED)
set(MORPHOVIEW_LIBRARIES morpho-lib glad glfw)

add_executable(morphoview ${sources})
target_include_directories(morphoview PUBLIC .)
target_link_libraries(morphoview PUBLIC ${MORPHOVIEW_LIBRARIES})
