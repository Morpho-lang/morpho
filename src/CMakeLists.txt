target_sources(morpho
    PRIVATE
        morpho.h build.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        morpho.h build.h
)

add_subdirectory(builtin)
add_subdirectory(classes)
add_subdirectory(core)
add_subdirectory(datastructures)
add_subdirectory(debug)
add_subdirectory(geometry)
add_subdirectory(linalg)
add_subdirectory(support)
