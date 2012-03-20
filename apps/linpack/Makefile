
ifndef LG_RT_DIR
$(error LG_RT_DIR variable is not defined, aborting build)
endif

#Flags for directing the runtime makefile what to include
DEBUG=1                   # Include debugging symbols
OUTPUT_LEVEL=LEVEL_DEBUG  # Compile time print level
SHARED_LOWLEVEL=1	  # Use the shared low level
#ALT_MAPPERS=1		  # Include the alternative mappers

# Put the binary file name here
OUTFILE		:= linpack
# List all the application source files here
GEN_SRC		:= linpack.cc	# .cc files
GEN_GPU_SRC	:=		# .cu files

# You can modify these variables, some will be appended to by the runtime makefile
INC_FLAGS	:=
CC_FLAGS	:=
NVCC_FLAGS	:=
GASNET_FLAGS	:=
LD_FLAGS	:=

ifdef SHARED_LOWLEVEL
	CC_FLAGS += -DUSE_LINPACK_SHARED
endif

###########################################################################
#
#   Don't change anything below here
#   
###########################################################################

# General shell commands
SHELL	:= /bin/sh
SH	:= sh
RM	:= rm -f
LS	:= ls
MKDIR	:= mkdir
MV	:= mv
CP	:= cp
SED	:= sed
ECHO	:= echo
TOUCH	:= touch
MAKE	:= make
ifndef GCC
GCC	:= g++
endif
ifndef NVCC
NVCC	:= nvcc
endif
SSH	:= ssh
SCP	:= scp

common_all : all

.PHONY	: common_all

# All these variables will be filled in by the runtime makefile
LOW_RUNTIME_SRC	:=
HIGH_RUNTIME_SRC:=
GPU_RUNTIME_SRC	:=
MAPPER_SRC	:=

include $(LG_RT_DIR)/runtime.mk

GEN_OBJS	:= $(GEN_SRC:.cc=.o)
GEN_GPU_OBJS	:= $(GEN_GPU_SRC:.cu=.o)
LOW_RUNTIME_OBJS:= $(LOW_RUNTIME_SRC:.cc=.o)
HIGH_RUNTIME_OBJS:=$(HIGH_RUNTIME_SRC:.cc=.o)
GPU_RUNTIME_OBJS:=$(GPU_RUNTIME_SRC:.cu=.o)
MAPPER_OBJS	:= $(MAPPER_SRC:.cc=.o)

ALL_OBJS	:= $(GEN_OBJS) $(GEN_GPU_OBJS) $(LOW_RUNTIME_OBJS) $(HIGH_RUNTIME_OBJS) $(GPU_RUNTIME_OBJS) $(MAPPER_OBJS)

all:
	$(MAKE) $(OUTFILE)

$(OUTFILE) : $(ALL_OBJS)
	@echo "---> Linking objects into one binary: $(OUTFILE)"
	$(GCC) -o $(OUTFILE) $(ALL_OBJS) $(LD_FLAGS) $(GASNET_FLAGS)

$(GEN_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(LOW_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(HIGH_RUNTIME_OBJS) : %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(MAPPER_OBJS)	: %.o : %.cc
	$(GCC) -o $@ -c $< $(INC_FLAGS) $(CC_FLAGS)

$(GEN_GPU_OBJS) : %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

$(GPU_RUNTIME_OBJS): %.o : %.cu
	$(NVCC) -o $@ -c $< $(INC_FLAGS) $(NVCC_FLAGS)

clean:
	@$(RM) -rf $(ALL_OBJS) $(OUTFILE)
