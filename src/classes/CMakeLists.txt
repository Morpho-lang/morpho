target_sources(morpho
    PRIVATE
        classes.h
        array.c       array.h
        closure.c     closure.h
        clss.c        clss.h
        cmplx.c       cmplx.h
        dict.c        dict.h
        err.c         err.h
        file.c        file.h
        flt.c         flt.h
        function.c    function.h
        instance.c    instance.h
        int.c         int.h
        invocation.c  invocation.h
        json.c        json.h
        list.c        list.h
        range.c       range.h
        strng.c       strng.h
        system.c      system.h
        upvalue.c     upvalue.h
)

target_sources(morpho
    INTERFACE
    FILE_SET public_headers
    TYPE HEADERS
    FILES
        classes.h
        array.h
        closure.h
        clss.h
        cmplx.h
        dict.h
        err.h
        file.h
        floatveneer.h
        function.h
        instance.h
        invocation.h
        json.h
        list.h
        range.h
        strng.h
        system.h
        upvalue.h
)
