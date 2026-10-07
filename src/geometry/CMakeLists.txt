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

add_subdirectory(functionals)
