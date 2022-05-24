src = $(wildcard *.c) ../morpho5/datastructures/varray.c ../deps/glad/src/glad.c ../morpho5/utils/memory.c
obj = $(src:.c=.o)
PREFIX = /usr/local

LDFLAGS  = -lm -lcblas -llapack -lglfw -lfreetype
CFLAGS   = -std=c99 -O3 -I/usr/local/include/freetype2 -I. -I../morpho5/datastructures -I../morpho5 -I../morpho5/utils -I../deps/glad/include

morphoview: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS) $(MODULES) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) morphoview

.PHONY: install
install: morphoview
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp $< $(DESTDIR)$(PREFIX)/bin/morphoview
	make clean
