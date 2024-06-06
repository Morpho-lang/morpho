target_sources(morpho
    PRIVATE
        field.c        field.h
        functional.c   functional.h
        integrate.c    integrate.h
        mesh.c         mesh.h
        selection.c    selection.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        field.h
        functional.h
        integrate.h
        mesh.h
        selection.h
)
