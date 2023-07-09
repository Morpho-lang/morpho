target_sources(morpho
    PRIVATE
        morpho.h build.h
)

add_subdirectory(builtin)
add_subdirectory(classes)
add_subdirectory(datastructures)
add_subdirectory(geometry)
add_subdirectory(linalg)
add_subdirectory(utils)
add_subdirectory(vm)
