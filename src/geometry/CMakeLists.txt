target_sources(morpho
    PRIVATE
        discretization.c    discretization.h
        field.c             field.h
        functional.c        functional.h
        geometry.c          geometry.h
        integrate.c         integrate.h
        mesh.c              mesh.h
        selection.c         selection.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        discretization.h
        field.h
        functional.h
        geometry.h
        integrate.h
        mesh.h
        selection.h
)
