target_sources(morpho
    PRIVATE
        common.c      common.h
        extensions.c  extensions.h
        lex.c         lex.h
        memory.c      memory.h
        parse.c       parse.h
        random.c      random.h
        resources.c   resources.h
        threadpool.c  threadpool.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        common.h
        extensions.h
        lex.h
        memory.h
        parse.h
        random.h
        resources.h
        threadpool.h
)
