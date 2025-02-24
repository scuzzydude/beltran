#CC=gcc
#NCCC=/usr/local/cuda-12.5/bin/nvcc
#CFLAGS=-I. -I../libaio/src -I./argtable -I./spdk/include -I../../include -I/usr/local/cuda-12.5/bin/../targets/x86_64-linux/include -I../../include/freestanding/include
#CFLAGS=-I. -I../libaio/src -I./argtable -I./spdk/include -I../../include -I/usr/local/cuda/include
#CFLAGS=-I. -I../libaio/src -I./argtable -I./spdk/include -I../../include  -I../../include/freestanding/include

#LIBS=-luring -pthread -lnvm
#LIBS=-luring -Xcompiler="-pthread"  
#LIBS=-luring -Xcompiler="-pthread" 
#LIBS=-luring -Xcompiler="-pthread"
#LIBS=-llibnvm.so -luring -Xcompiler="-pthread" 
#NVLIBS=-lnvm

#CFLAGS += -g -Wall -O2 -D_GNU_SOURCE -lm -L../liburing/src/ -laio
#CFLAGS += -Wall -O2 -D_GNU_SOURCE -laio -lm -L../liburing/src/ -L../libaio/src 
#CFLAGS += -O2 -DNVIDIA_BAM -D_GNU_SOURCE -laio -lm -L../liburing/src/ -L../libaio/src -L../../build/lib
#CFLAGS += -g -O2 -DNVIDIA_BAM -D_GNU_SOURCE -laio -lm -L../liburing/src/ -L../libaio/src -L../../build/lib
#CUDAFLAGS=-arch=sm_70 -include "cuda_runtime.h" -D__NV_LEGACY_LAUNCH

#CFLAGS += -DASTROS_SPDK

#%.o: %.cu
#	$(NCCC) -dc $(CUDAFLAGS) $< -o $@

#all:
	
#	$(CC) -o astros astros.c astros_linux_aio.c astros_batters.c astros_inning.c astros_ccb.c astros_lineup.c astros_scorer.c astros_signs.c astros_sync_batters.c argtable3.c astros_linux.c astros_spdk.c astros_bam_host.c astros_bam.cu $(CFLAGS) $(LIBS)
#	$(CC) -o astros astros.c astros_linux_aio.c astros_batters.c astros_inning.c astros_ccb.c astros_lineup.c astros_scorer.c astros_signs.c astros_sync_batters.c argtable3.c astros_linux.c astros_spdk.c astros_bam_host.c  $(CFLAGS) $(LIBS)

#astros: astros.o astros_linux_aio.o astros_batters.o astros_inning.o astros_ccb.o astros_lineup.o astros_scorer.o astros_signs.o astros_sync_batters.o argtable3.o astros_linux.o astros_spdk.o astros_bam_host.o  
#	$(CC) -o astros astros.o astros_linux_aio.o astros_batters.o astros_inning.o astros_ccb.o astros_lineup.o astros_scorer.o astros_signs.o astros_sync_batters.o argtable3.o astros_linux.o astros_spdk.o astros_bam_host.o  $(CFLAGS) $(LIBS)

#	cp astros ../../../run

#astros_bam.o: astros_bam.cu
#	$(CC) -c astros_bam.cu -o astros_bam.o -arch=sm_70 $(CFLAGS) $(LIBS) -include "cuda_runtime.h" -D__NV_LEGACY_LAUNCH 
#	$(CC) -c astros_bam.cu -o astros_bam.o -arch=sm_70 $(CFLAGS) $(LIBS) -include "cuda_runtime.h" -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ -D__NVCC__  -std=c++11
#	$(CC) -c astros_bam.cu -o astros_bam.o -arch=sm_70 $(CFLAGS) $(LIBS) -include "cuda_runtime.h" -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ -D__NVCC__  -std=c++11
#	$(NCCC) -c astros_bam.cu -o astros_bam.o -arch=sm_70 $(CFLAGS) $(NVLIBS) 
	



#https://bikulov.org/blog/2013/09/02/universal-simple-makefile/ 





APPNAME := astros
SOURCES := $(wildcard *.cu *.c *.cpp)
#SOURCES := $(wildcard *.c)

OBJECTS := $(patsubst src%,obj%, $(patsubst %.cu,%.device.o, $(patsubst %.cpp,%.o, $(patsubst %.c,%.o, $(SOURCES)))))

INCLUDE := -I. -I../libaio/src -I./argtable -I./spdk/include -I../../include -I/usr/local/cuda-12.5/bin/../targets/x86_64-linux/include -I../../include/freestanding/include
LIBPATH := -L../liburing/src/ -L../libaio/src -L../../build/lib
LIBS    := -luring -laio -lm -lcudart -lnvm
#LIBS    := -laio -lm -lcudart -lnvm

#FLAGS    := -g -O2 -DNVIDIA_BAM -D_GNU_SOURCE  
FLAGS    := -DNVIDIA_BAM -D_GNU_SOURCE -D__CUDACC__ -D__CUDA__ -O3 -std=c++11 -DNVCC
CCFLAGS  := $(FLAGS)
CXXFLAGS := $(FLAGS)

#GENCODE_FLAGS := -arch=sm_70
#GENCODE_FLAGS := -gencode arch=compute_70,code=sm_70 
#GENCODE_FLAGS := -gencode arch=compute_70,code=sm_70 
#NVCCFLAGS     := $(GENCODE_FLAGS) -include "cuda_runtime.h"  -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__ -D__NVCC__  
GENCODE_FLAGS := -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90

#-ewp added to avoid segmentation fault when linking to CUDA code, even though CUDA code not called.
NVCCFLAGS     := $(GENCODE_FLAGS) 

CC   := /usr/local/cuda-12.5/bin/nvcc
CXX  := /usr/local/cuda-12.5/bin/nvcc
NVCC := /usr/local/cuda-12.5/bin/nvcc

all: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(OBJECTS) -o $(APPNAME) $(LIBPATH) $(LIBS) 

	cp astros ../../../run

%.o: ./%.c
	$(CC) $(CCFLAGS) $(INCLUDE) -c $< -o $@


%.device.o: ./%.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE) -c $< -o $@




















clean:
	rm astros *.o


     
