src = $(wildcard *.c) $(wildcard */*.c) 
obj = $(src:.c=.o) 

cudasrc = $(wildcard *.c) $(wildcard */*.c) $(wildcard */*/*.cu) $(wildcard */*/*.c) 
cudatmp = $(cudasrc:.c=.o)
cudaobj = $(cudatmp:.cu=.o) 

PREFIX = /usr/local
HELPDIR = /usr/local/share/morpho/help
MODULESDIR = /usr/local/share/morpho/modules
help = $(wildcard docs/*.md)
modules = $(wildcard modules/*)
NVCC = nvcc

LDFLAGS  = -lm -lblas -llapacke -lcxsparse -lcublas -lcusolver -lcudart 
CFLAGS   = -std=c99 -O3 -I. -I/usr/include/suitesparse -I./datastructures  -I./datastructures/GPUdatastructures -I./geometry -I./interface -I./utils -I./vm -I./builtin
CUDAFLAGS = -rdc=true -O3 -I. -I/usr/include/suitesparse -I./datastructures  -I./datastructures/GPUdatastructures -I./geometry -I./interface -I./utils -I./vm -I./builtin

%.o: %.cu
	$(NVCC) -c -o $@ $< $(LDFLAGS) $(CUDAFLAGS) 

cudainterface.o: kernal.h

morpho5cuda: $(cudaobj)
	$(NVCC) -o $@ $^ $(LDFLAGS) $(MODULES) $(CUDAFLAGS) 

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

cudainstall: CFLAGS+= -DCUDA_ACC
cudainstall: morpho5cuda
	mkdir -p $(DESTDIR)$(PREFIX)/bin
	cp $< $(DESTDIR)$(PREFIX)/bin/morpho
	make modules
	make help
	make clean