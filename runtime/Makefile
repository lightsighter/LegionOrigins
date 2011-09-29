GASNET = /usr/local/gasnet-1.16.2

CCFLAGS = -I$(GASNET)/include -ggdb -Wall

LDFLAGS = -L$(GASNET)/lib -lrt

CCFLAGS += -DGASNET_CONDUIT_IBV -I$(GASNET)/include/ibv-conduit -DDEBUG_LOW_LEVEL #-DDEBUG_PRINT
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

hello.o : lowlevel.h

ring:	ring_test.o shared_lowlevel.o
	g++ -o ring $(CCFLAGS) $^ -lpthread

clean:
	rm -f ring *.o
