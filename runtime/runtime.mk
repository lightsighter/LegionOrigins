
GASNET 	:= /usr/local/gasnet-1.16.2
CUDA	:= /usr/local/cuda

ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

INC_FLAGS	+= -I$(LG_RT_DIR)
LD_FLAGS	+= -lrt -lpthread
ifndef SHARED_LOWLEVEL
INC_FLAGS 	+= -I$(GASNET)/include -I$(GASNET)/include/ibv-conduit 
INC_FLAGS	+= -I$(CUDA)/include
CC_FLAGS	+= -DGASNET_CONDUIT_IBV
LD_FLAGS	+= -L$(GASNET)/lib -lgasnet-ibv-par -libverbs
LD_LFAGS	+= -L$(CUDA)/lib64 -lcudart -lcuda -Wl,-rpath=$(CUDA)/lib64
NVCC_FLAGS	+= -arch=sm_20
ifdef DEBUG
NVCC_FLAGS	+= -g -G
else
NVCC_FLAGS	+= -O3
endif
endif

ifdef DEBUG
CC_FLAGS	+= -DDEBUG_LOW_LEVEL -DDEBUG_HIGH_LEVEL -ggdb -Wall
else
CC_FLAGS	+= -O3 -DCOMPILER_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)
endif

# Set the source files
ifndef SHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/lowlevel.cc $(LG_RT_DIR)/lowlevel_gpu.cc $(LG_RT_DIR)/default_mapper.cc
GPU_RUNTIME_SRC += 
else
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/shared_lowlevel.cc $(LG_RT_DIR)/shared_mapper.cc
endif

HIGH_RUNTIME_SRC += $(LG_RT_DIR)/legion.cc
