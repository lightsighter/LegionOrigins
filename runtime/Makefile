GASNET = /usr/local/gasnet-1.16.2

CCFLAGS = -I$(GASNET)/include -ggdb -Wall
#CCFLAGS = -I$(GASNET)/include -O3 -Wall

LDFLAGS = -L$(GASNET)/lib -lrt

CCFLAGS += -DGASNET_CONDUIT_IBV -I$(GASNET)/include/ibv-conduit -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL #-DDEBUG_PRINT_HIGH_LEVEL #-DDEBUG_PRINT
LDFLAGS += -lgasnet-ibv-par -libverbs

%.o : %.cc
	g++ $(CCFLAGS) -c $< -o $@

foo : lowlevel.o
	g++ $(CCFLAGS) $^ $(LDFLAGS) -o $@

hello : hello.o lowlevel.o
	g++ $(CCFLAGS) $^ $(LDFLAGS) -o $@

gas_simple : gas_simple.c
	gcc $(CCFLAGS) gas_simple.c -o gas_simple $(LDFLAGS)

lowlevel.o : lowlevel.h activemsg.h

shared_lowlevel.o : lowlevel.h

highlevel.o : highlevel.h lowlevel.h

hello.o : lowlevel.h

ring_test.o : lowlevel.h

tree_test.o : highlevel.h

circuit.o : highlevel.h

ring2:	ring_test.o lowlevel.o
	g++ -o $@ $(CCFLAGS) $^ $(LDFLAGS)

ring:	ring_test.o shared_lowlevel.o
	g++ -o $@ $(CCFLAGS) $^ -lpthread

tree:	shared_lowlevel.o highlevel.o tree_test.o
	g++ -o $@ $(CCFLAGS) $^ -lpthread

tree2:	lowlevel.o highlevel.o tree_test.o
	g++ -o $@ $(CCFLAGS) $^ $(LDFLAGS)

ckt:	shared_lowlevel.o highlevel.o circuit.o
	g++ -o $@ $(CCFLAGS) $^ -lpthread

ckt2:	lowlevel.o highlevel.o circuit.o
	g++ -o $@ $(CCFLAGS) $^ $(LDFLAGS)

clean:
	rm -f ckt ring tree *.o
