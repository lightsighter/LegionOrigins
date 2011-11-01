GASNET = /usr/local/gasnet-1.16.2

CCFLAGS = -I$(GASNET)/include
#CCFLAGS = -I$(GASNET)/include -O3 -Wall

GCC = gcc
GCCFLAGS = $(CCFLAGS) -ggdb -Wall

LDFLAGS = -L$(GASNET)/lib -lrt

CCFLAGS += -DGASNET_CONDUIT_IBV -I$(GASNET)/include/ibv-conduit -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL #-DDEBUG_PRINT_HIGH_LEVEL #-DDEBUG_PRINT
LDFLAGS += -lgasnet-ibv-par -libverbs

CCFLAGS += -I/usr/local/cuda/include
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -g $(CCFLAGS)
LDFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcuda -Wl,-rpath=/usr/local/cuda/lib64

%.o : %.cc
	g++ $(GCCFLAGS) -c $< -o $@

%.o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

foo : lowlevel.o
	g++ $(GCCFLAGS) $^ $(LDFLAGS) -o $@

hello : hello.o lowlevel.o
	g++ $(GCCFLAGS) $^ $(LDFLAGS) -o $@

gas_simple : gas_simple.c
	gcc $(GCCFLAGS) gas_simple.c -o gas_simple $(LDFLAGS)

lowlevel_gpu.o : lowlevel_gpu.h lowlevel_impl.h lowlevel.h activemsg.h

gputest.o : lowlevel.h

lowlevel.o : lowlevel.h lowlevel_impl.h activemsg.h

shared_lowlevel.o : lowlevel.h

highlevel.o : highlevel.h lowlevel.h

hello.o : lowlevel.h

ring_test.o : lowlevel.h

tree_test.o : highlevel.h

circuit.o : highlevel.h

gputest:	gputest.o lowlevel_gpu.o lowlevel.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

ring2:	ring_test.o lowlevel_gpu.o lowlevel.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

ring:	ring_test.o shared_lowlevel.o
	g++ -o $@ $(GCCFLAGS) $^ -lpthread

tree:	shared_lowlevel.o highlevel.o tree_test.o
	g++ -o $@ $(GCCFLAGS) $^ -lpthread

tree2:	lowlevel.o highlevel.o tree_test.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

ckt:	shared_lowlevel.o highlevel.o circuit.o
	g++ -o $@ $(GCCFLAGS) $^ -lpthread

ckt2:	lowlevel.o highlevel.o circuit.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

clean:
	rm -f ring tree *.o
