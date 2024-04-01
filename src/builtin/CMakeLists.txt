target_sources(morpho
    PRIVATE
        builtin.c       builtin.h     
        functiondefs.c  functiondefs.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        builtin.h functiondefs.h
)
