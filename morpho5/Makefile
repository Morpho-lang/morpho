src = $(wildcard *.c) $(wildcard */*.c)
obj = $(src:.c=.o)
PREFIX = /usr/local
HELPDIR = /usr/local/share/morpho/help
MODULESDIR = /usr/local/share/morpho/modules
help = $(wildcard docs/*.md)
modules = $(wildcard modules/*)

LDFLAGS  = -lm -lcblas -llapack -lcxsparse
CFLAGS   = -std=c99 -O3 -I. -I./datastructures -I./geometry -I./interface -I./utils -I./vm -I./builtin

morpho5: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS) $(MODULES) $(CFLAGS)

.PHONY: clean
clean:
	rm -f $(obj) morpho5

.PHONY: help
help:
	mkdir -p $(HELPDIR)
	cp $(help) $(HELPDIR)

.PHONY: modules
modules:
	mkdir -p $(MODULESDIR)
	cp -R $(modules) $(MODULESDIR)

.PHONY: install
install: morpho5
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp $< $(DESTDIR)$(PREFIX)/bin/morpho5
	make modules
	make help
	make clean
