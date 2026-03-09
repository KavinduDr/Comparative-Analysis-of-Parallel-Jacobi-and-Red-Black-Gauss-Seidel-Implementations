

















CC       = gcc-14
MPICC    = OMPI_CC=gcc-14 MPICH_CC=gcc-14 mpicc
NVCC     = nvcc


CFLAGS   = -O2 -Wall -Wextra -I include
LDFLAGS  = -lm
OMP_FLAG = -fopenmp
PTH_FLAG = -lpthread
NVFLAGS  = -O2 -arch=sm_50


BIN = bin


SERIAL   = $(BIN)/serial
OPENMP   = $(BIN)/openmp_solver
PTHREADS = $(BIN)/pthreads_solver
MPI      = $(BIN)/mpi_solver
HYBRID   = $(BIN)/hybrid_solver
CUDA     = $(BIN)/cuda_solver

.PHONY: all all_cpu serial openmp pthreads mpi hybrid cuda clean run_all


all_cpu: serial openmp pthreads

all: all_cpu cuda

$(BIN):
	mkdir -p $(BIN)


serial: $(SERIAL)
$(SERIAL): src/serial.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ src/serial.c $(LDFLAGS)


openmp: $(OPENMP)
$(OPENMP): src/openmp_solver.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ src/openmp_solver.c $(LDFLAGS)


pthreads: $(PTHREADS)
$(PTHREADS): src/pthreads_solver.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ src/pthreads_solver.c $(LDFLAGS) $(PTH_FLAG)


mpi: $(MPI)
$(MPI): src/mpi_solver.c include/common.h | $(BIN)
	$(MPICC) $(CFLAGS) -o $@ src/mpi_solver.c $(LDFLAGS)


hybrid: $(HYBRID)
$(HYBRID): src/hybrid_solver.c include/common.h | $(BIN)
	$(MPICC) $(CFLAGS) $(OMP_FLAG) -o $@ src/hybrid_solver.c $(LDFLAGS)


cuda: $(CUDA)
$(CUDA): src/cuda_solver.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ src/cuda_solver.cu -lm


clean:
	rm -rf $(BIN)


GRID  ?= 200
ITERS ?= 5000
TOL   ?= 1e-6

run_all: all_cpu
	@echo ""
	@echo "===== Running Serial ====="
	$(SERIAL) $(GRID) $(ITERS) $(TOL)
	@echo ""
	@echo "===== Running OpenMP (2 threads) ====="
	$(OPENMP) $(GRID) $(ITERS) $(TOL) 2
	@echo "===== Running OpenMP (4 threads) ====="
	$(OPENMP) $(GRID) $(ITERS) $(TOL) 4
	@echo ""
	@echo "===== Running Pthreads (2 threads) ====="
	$(PTHREADS) $(GRID) $(ITERS) $(TOL) 2
	@echo "===== Running Pthreads (4 threads) ====="
	$(PTHREADS) $(GRID) $(ITERS) $(TOL) 4
	@echo ""
	@echo "All benchmarks complete."
