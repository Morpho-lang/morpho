target_sources(morpho
    PRIVATE
        compile.c    compile.h
        gc.c         gc.h
        vm.c         vm.h
        core.h
        opcodes.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        compile.h
        gc.h
        optimize.h
        vm.h
        core.h
        opcodes.h
)
