src = $(wildcard *.c) $(wildcard */*.c)
obj = $(src:.c=.o)

ifdef MORPHORESOURCESDIR
RESOURCEPREFIX = $(MORPHORESOURCESDIR)
HELPDIR = $(RESOURCEPREFIX)/share/help
MODULESDIR = $(RESOURCEPREFIX)/share/modules
else 
RESOURCEPREFIX = /usr/local
HELPDIR = $(RESOURCEPREFIX)/share/morpho/help
MODULESDIR = $(RESOURCEPREFIX)/share/morpho/modules
endif 

ifdef DESTDIR 
INSTALLDIR = $(DESTDIR)
else 
INSTALLDIR = /usr/local/bin
endif 

help = $(wildcard docs/*.md)
modules = $(wildcard modules/*)

LDFLAGS  = -lm -lcblas -llapack -lcxsparse
CFLAGS   = $(EXTCFLAGS) -DMORPHO_RESOURCESDIR=\"$(RESOURCEPREFIX)\" -std=c99 -O3 -I. -I./datastructures -I./geometry -I./interface -I./utils -I./vm -I./builtin

morpho5: $(obj)
	$(CC) -o $@ $^ $(LDFLAGS) $(CFLAGS)

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
	mkdir -p $(INSTALLDIR)
	cp $< $(INSTALLDIR)/morpho5
	make modules
	make help
	make clean
	