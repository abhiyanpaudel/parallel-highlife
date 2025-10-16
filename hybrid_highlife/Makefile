# Compact Makefile for CUDA, MPI, and Hybrid builds

# Toolchain
NVCC    ?= nvcc
MPICC   ?= mpicc

# Flags
CFLAGS  ?= -O3 
GPU     ?= 70
NVFLAGS ?= -O3 -gencode arch=compute_$(GPU),code=sm_$(GPU)

# CUDA libs
CUDA_LIB ?= /usr/local/cuda/lib64
LDCUDA    = -L$(CUDA_LIB) -lcudart -lcudadevrt -lstdc++

# Targets
.PHONY: all cuda mpi hybrid clean
all: cuda mpi hybrid

# --- CUDA  ---
cuda: highlife_cuda
highlife: highlife_cuda.cu
	$(NVCC) $(NVFLAGS) -o $@ $^

# --- MPI  ---
mpi: highlife_mpi
highlife_mpi: highlife_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $^

# --- Hybrid ---
hybrid: highlife_exe
highlife_mpi.o: highlife_mpi.c
	$(MPICC) $(CFLAGS) -c $< -o $@
highlife_cuda.o: highlife_cuda.cu
	$(NVCC)  $(NVFLAGS) -c $< -o $@
highlife_exe: highlife_mpi.o highlife_cuda.o
	$(MPICC) $(CFLAGS) $^ -o $@ $(LDCUDA)

clean:
	rm -f highlife_cuda highlife_mpi highlife_exe *.o
