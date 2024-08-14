target_sources(morpho
    PRIVATE
        newlinalg.c  newlinalg.h
        complexmatrix.c complexmatrix.h
        matrix.c  matrix.h
        sparse.c  sparse.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        newlinalg.h
        complexmatrix.h
        matrix.h
        sparse.h
)
