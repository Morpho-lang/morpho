target_sources(morpho
    PRIVATE
        dictionary.c    dictionary.h
        error.c         error.h
        object.c        object.h
        program.c       program.h
        syntaxtree.c    syntaxtree.h
        value.c         value.h
        varray.c        varray.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        dictionary.h
        error.h
        object.h
        program.h
        syntaxtree.h
        value.h
        varray.h
)
