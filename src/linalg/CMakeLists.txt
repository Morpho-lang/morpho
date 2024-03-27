target_sources(morpho
    PRIVATE
        matrix.c  matrix.h
        sparse.c  sparse.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        matrix.h
        sparse.h
)
