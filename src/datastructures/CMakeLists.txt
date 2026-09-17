target_sources(morpho
    PRIVATE
        dictionary.c        dictionary.h
        debugannotation.c   debugannotation.h
        error.c             error.h
        object.c            object.h
        objectpool.c        objectpool.h
        program.c           program.h
        signature.c         signature.h
        syntaxtree.c        syntaxtree.h
        value.c             value.h
        varray.c            varray.h
        version.c           version.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        dictionary.h
        debugannotation.h
        error.h
        object.h
        objectpool.h
        program.h
        signature.h
        syntaxtree.h
        value.h
        varray.h
        version.h
)
