GASNET = /usr/local/gasnet-1.16.2

CCFLAGS = -I$(GASNET)/include -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL #-DDEBUG_PRINT_HIGH_LEVEL #-DDEBUG_PRINT
#CCFLAGS = -I$(GASNET)/include -O3
#CCFLAGS = -I$(GASNET)/include -O3 -DCOMPILE_TIME_MIN_LEVEL=LEVEL_NONE

# for circuit
#CCFLAGS = -I$(GASNET)/include -O3 -DCOMPILE_TIME_MIN_LEVEL=LEVEL_NONE

# for fluid
#CCFLAGS = -I$(GASNET)/include -O3 -DCOMPILE_TIME_MIN_LEVEL=LEVEL_NONE -DUSE_MASKED_COPIES

GCC = gcc
GCCFLAGS = $(CCFLAGS) -ggdb -Wall

LDFLAGS = -L$(GASNET)/lib -lrt

CCFLAGS += -DGASNET_CONDUIT_IBV -I$(GASNET)/include/ibv-conduit
LDFLAGS += -lgasnet-ibv-par -libverbs

CCFLAGS += -I/usr/local/cuda/include
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -g $(CCFLAGS) -arch=sm_20
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

circuit_gpu.o : highlevel.h lowlevel.h

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

ckt2:	lowlevel.o lowlevel_gpu.o highlevel.o circuit.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

cktgpu2:	lowlevel.o lowlevel_gpu.o highlevel.o circuit_gpu.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

fluid:	shared_lowlevel.o highlevel.o fluid.o
	g++ -o $@ $(GCCFLAGS) $^ -lpthread

fluid2:	lowlevel.o lowlevel_gpu.o highlevel.o fluid.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

srch2:	lowlevel.o lowlevel_gpu.o highlevel.o search.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

saxpy:  lowlevel.o lowlevel_gpu.o highlevel.o saxpy.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

saxpy2:	shared_lowlevel.o highlevel.o saxpy.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

saxpy_gpu:	lowlevel.o lowlevel_gpu.o highlevel.o saxpy_gpu.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

clean:
	rm -f fluid fluid2 ckt ckt2 cktgpu2 ring tree saxpy saxpy_gpu *.o
