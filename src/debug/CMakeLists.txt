target_sources(morpho
    PRIVATE
        debug.c      debug.h
        profile.c    profile.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        debug.h
        profile.h
)
