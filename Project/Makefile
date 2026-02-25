# ============================================================================
# Makefile for HPC Project: Parallel Jacobi & Red-Black Gauss-Seidel Solvers
# ============================================================================
#
# Targets:
#   make serial        - Build serial version
#   make openmp        - Build OpenMP version
#   make pthreads      - Build Pthreads version
#   make mpi           - Build MPI version
#   make hybrid        - Build Hybrid MPI+OpenMP version
#   make cuda          - Build CUDA version
#   make all_cpu       - Build all CPU versions (serial, openmp, pthreads, mpi, hybrid)
#   make all           - Build everything including CUDA
#   make clean         - Remove all binaries
#   make run_all       - Run all benchmarks
# ============================================================================

# Compilers
CC       = gcc
MPICC    = mpicc
NVCC     = nvcc

# Flags
CFLAGS   = -O2 -Wall -Wextra -I include
LDFLAGS  = -lm
OMP_FLAG = -fopenmp
PTH_FLAG = -lpthread
NVFLAGS  = -O2 -arch=sm_50

# Output directory
BIN = bin

# Targets
SERIAL   = $(BIN)/serial
OPENMP   = $(BIN)/openmp_solver
PTHREADS = $(BIN)/pthreads_solver
MPI      = $(BIN)/mpi_solver
HYBRID   = $(BIN)/hybrid_solver
CUDA     = $(BIN)/cuda_solver

.PHONY: all all_cpu serial openmp pthreads mpi hybrid cuda clean run_all

# Default: build all CPU targets
all_cpu: serial openmp pthreads mpi hybrid

all: all_cpu cuda

$(BIN):
	mkdir -p $(BIN)

# ----- Serial -----
serial: $(SERIAL)
$(SERIAL): src/serial.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ src/serial.c $(LDFLAGS)

# ----- OpenMP -----
openmp: $(OPENMP)
$(OPENMP): src/openmp_solver.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ src/openmp_solver.c $(LDFLAGS)

# ----- Pthreads -----
pthreads: $(PTHREADS)
$(PTHREADS): src/pthreads_solver.c include/common.h | $(BIN)
	$(CC) $(CFLAGS) -o $@ src/pthreads_solver.c $(LDFLAGS) $(PTH_FLAG)

# ----- MPI -----
mpi: $(MPI)
$(MPI): src/mpi_solver.c include/common.h | $(BIN)
	$(MPICC) $(CFLAGS) -o $@ src/mpi_solver.c $(LDFLAGS)

# ----- Hybrid MPI+OpenMP -----
hybrid: $(HYBRID)
$(HYBRID): src/hybrid_solver.c include/common.h | $(BIN)
	$(MPICC) $(CFLAGS) $(OMP_FLAG) -o $@ src/hybrid_solver.c $(LDFLAGS)

# ----- CUDA -----
cuda: $(CUDA)
$(CUDA): src/cuda_solver.cu | $(BIN)
	$(NVCC) $(NVFLAGS) -o $@ src/cuda_solver.cu -lm

# ----- Clean -----
clean:
	rm -rf $(BIN)

# ----- Run all benchmarks -----
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
	@echo "===== Running MPI (2 processes) ====="
	mpirun --oversubscribe -np 2 $(MPI) $(GRID) $(ITERS) $(TOL)
	@echo "===== Running MPI (4 processes) ====="
	mpirun --oversubscribe -np 4 $(MPI) $(GRID) $(ITERS) $(TOL)
	@echo ""
	@echo "===== Running Hybrid (2 procs x 2 threads) ====="
	mpirun --oversubscribe -np 2 $(HYBRID) $(GRID) $(ITERS) $(TOL) 2
	@echo ""
	@echo "All benchmarks complete."
