
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
ifdef SHARED_LOWLEVEL
LD_FLAGS	+= -L$(CUDA)/lib64 -lcudart -lcuda -Wl,-rpath=$(CUDA)/lib64
else
LD_FLAGS	+= -L$(CUDA)/lib64 -lcudart -lcuda -Xlinker -rpath=$(CUDA)/lib64
endif
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
CC_FLAGS	+= -O3 
endif

# Manage the output setting
CC_FLAGS	+= -DCOMPILE_TIME_MIN_LEVEL=$(OUTPUT_LEVEL)

#CC_FLAGS += -DUSE_MASKED_COPIES

# Set the source files
ifndef SHARED_LOWLEVEL
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/lowlevel.cc $(LG_RT_DIR)/lowlevel_gpu.cc 
GPU_RUNTIME_SRC += 
else
LOW_RUNTIME_SRC	+= $(LG_RT_DIR)/shared_lowlevel.cc 
endif

# If you want to go back to using the shared mapper, comment out the next line
# and uncomment the one after that
MAPPER_SRC	+= $(LG_RT_DIR)/default_mapper.cc
#MAPPER_SRC	+= $(LG_RT_DIR)/shared_mapper.cc
ifdef ALT_MAPPERS
MAPPER_SRC	+= $(LG_RT_DIR)/alt_mappers.cc
endif

HIGH_RUNTIME_SRC += $(LG_RT_DIR)/legion.cc 
