LG_RT_DIR = ../../runtime
GASNET = /usr/local/gasnet-1.16.2
CCFLAGS = -I$(LG_RT_DIR) -I$(GASNET)/include -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL

LDFLAGS = -L$(GASNET)/lib -lrt

CCFLAGS += -DGASNET_CONDUIT_IBV -I$(GASNET)/include/ibv-conduit
LDFLAGS += -lgasnet-ibv-par -libverbs

CCFLAGS += -I/usr/local/cuda/include
NVCC = /usr/local/cuda/bin/nvcc
NVCCFLAGS = -g $(CCFLAGS) -arch=sm_20
LDFLAGS += -L/usr/local/cuda/lib64 -lcudart -lcuda -Wl,-rpath=/usr/local/cuda/lib64

GCC = gcc
GCCFLAGS = $(CCFLAGS) -ggdb -Wall

saxpy : saxpy.o lowlevel.o lowlevel_gpu.o legion.o
	g++ -o $@ $(GCCFLAGS) $^ $(LDFLAGS)

saxpy.o : saxpy.cc
	g++ $(GCCFLAGS) -c $< -o $@

legion.o : $(LG_RT_DIR)/legion.h $(LG_RT_DIR)/lowlevel.h
	g++ $(GCCFLAGS) -c $< -o $@

lowlevel_gpu.o : $(LG_RT_DIR)/lowlevel_gpu.cc $(LG_RT_DIR)/lowlevel_gpu.h $(LG_RT_DIR)/lowlevel_impl.h $(LG_RT_DIR)/lowlevel.h $(LG_RT_DIR)/activemsg.h
	g++ $(GCCFLAGS) -c $< -o $@

lowlevel.o : $(LG_RT_DIR)/lowlevel.cc $(LG_RT_DIR)/lowlevel.h $(LG_RT_DIR)/lowlevel_impl.h $(LG_RT_DIR)/activemsg.h
	g++ $(GCCFLAGS) -c $< -o $@

clean :
	rm -f saxpy *.o
