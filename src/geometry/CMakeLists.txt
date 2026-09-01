target_sources(morpho
    PRIVATE
        fespace.c           fespace.h
        elements.c          elements.h
        field.c             field.h
        quadrature.c
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
        fespace.h
        elements.h
        field.h
        functional.h
        geometry.h
        integrate.h
        mesh.h
        selection.h
)

add_subdirectory(functionals)
