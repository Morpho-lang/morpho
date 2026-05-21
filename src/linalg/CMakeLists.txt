target_sources(morpho
    PRIVATE
        complexmatrix.c complexmatrix.h
        linalg.c  linalg.h
        matrix.c  matrix.h
        sparse.c  sparse.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        linalg.h
        matrix.h
        complexmatrix.h
        sparse.h
)
