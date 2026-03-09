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
CC       = gcc      # GNU C compiler for serial, OpenMP, and Pthreads builds
MPICC    = mpicc    # MPI C compiler wrapper (uses underlying CC with MPI flags)
NVCC     = nvcc     # NVIDIA CUDA compiler for GPU code

# Flags
CFLAGS   = -O2 -Wall -Wextra -I include  # Optimization level 2, enable all+extra warnings, add include path
LDFLAGS  = -lm                            # Link the math library (required for sin, sqrt, fabs, etc.)
OMP_FLAG = -fopenmp                       # Enable OpenMP support in GCC
PTH_FLAG = -lpthread                      # Link POSIX threads library
NVFLAGS  = -O2 -arch=sm_50               # CUDA: optimization level 2, target GPU compute capability 5.0

# Output directory
BIN = bin  # Directory where compiled binaries are placed

# Targets - define the output binary paths
SERIAL   = $(BIN)/serial            # Serial implementation binary
OPENMP   = $(BIN)/openmp_solver     # OpenMP implementation binary
PTHREADS = $(BIN)/pthreads_solver   # Pthreads implementation binary
MPI      = $(BIN)/mpi_solver        # MPI implementation binary
HYBRID   = $(BIN)/hybrid_solver     # Hybrid MPI+OpenMP implementation binary
CUDA     = $(BIN)/cuda_solver       # CUDA GPU implementation binary

.PHONY: all all_cpu serial openmp pthreads mpi hybrid cuda clean run_all  # Declare targets that are not actual files

# Default: build all CPU targets
all_cpu: serial openmp pthreads mpi hybrid  # Build all CPU-based implementations

all: all_cpu cuda  # Build everything: all CPU versions plus CUDA

$(BIN):            # Rule to create the bin directory if it doesn't exist
	mkdir -p $(BIN)  # Create bin directory (and parent dirs if needed)

# ----- Serial -----
serial: $(SERIAL)                              # Target alias for serial build
$(SERIAL): src/serial.c include/common.h | $(BIN)  # Build serial: depends on source and header, bin dir must exist
	$(CC) $(CFLAGS) -o $@ src/serial.c $(LDFLAGS)  # Compile serial.c with standard flags and link math library

# ----- OpenMP -----
openmp: $(OPENMP)                                        # Target alias for OpenMP build
$(OPENMP): src/openmp_solver.c include/common.h | $(BIN)  # Build OpenMP: depends on source, header, and bin dir
	$(CC) $(CFLAGS) $(OMP_FLAG) -o $@ src/openmp_solver.c $(LDFLAGS)  # Compile with OpenMP flag (-fopenmp) enabled

# ----- Pthreads -----
pthreads: $(PTHREADS)                                          # Target alias for Pthreads build
$(PTHREADS): src/pthreads_solver.c include/common.h | $(BIN)   # Build Pthreads: depends on source, header, and bin dir
	$(CC) $(CFLAGS) -o $@ src/pthreads_solver.c $(LDFLAGS) $(PTH_FLAG)  # Compile and link with pthread library

# ----- MPI -----
mpi: $(MPI)                                          # Target alias for MPI build
$(MPI): src/mpi_solver.c include/common.h | $(BIN)   # Build MPI: depends on source, header, and bin dir
	$(MPICC) $(CFLAGS) -o $@ src/mpi_solver.c $(LDFLAGS)  # Compile with MPI compiler wrapper (handles MPI includes/libs)

# ----- Hybrid MPI+OpenMP -----
hybrid: $(HYBRID)                                          # Target alias for Hybrid build
$(HYBRID): src/hybrid_solver.c include/common.h | $(BIN)   # Build Hybrid: depends on source, header, and bin dir
	$(MPICC) $(CFLAGS) $(OMP_FLAG) -o $@ src/hybrid_solver.c $(LDFLAGS)  # Compile with MPI compiler and OpenMP flag

# ----- CUDA -----
cuda: $(CUDA)                              # Target alias for CUDA build
$(CUDA): src/cuda_solver.cu | $(BIN)       # Build CUDA: depends on source file and bin dir
	$(NVCC) $(NVFLAGS) -o $@ src/cuda_solver.cu -lm  # Compile CUDA source with nvcc and link math library

# ----- Clean -----
clean:             # Remove all compiled binaries
	rm -rf $(BIN)  # Recursively delete the bin directory and all contents

# ----- Run all benchmarks -----
GRID  ?= 200    # Default grid size for benchmarks (overridable: make run_all GRID=500)
ITERS ?= 5000   # Default max iterations for benchmarks (overridable)
TOL   ?= 1e-6   # Default convergence tolerance for benchmarks (overridable)

run_all: all_cpu                                          # Run benchmarks: first build all CPU targets
	@echo ""                                               # Print blank line for readability
	@echo "===== Running Serial ====="                     # Print section header
	$(SERIAL) $(GRID) $(ITERS) $(TOL)                      # Run serial solver with benchmark parameters
	@echo ""                                               # Print blank line
	@echo "===== Running OpenMP (2 threads) ====="         # Print section header
	$(OPENMP) $(GRID) $(ITERS) $(TOL) 2                    # Run OpenMP with 2 threads
	@echo "===== Running OpenMP (4 threads) ====="         # Print section header
	$(OPENMP) $(GRID) $(ITERS) $(TOL) 4                    # Run OpenMP with 4 threads
	@echo ""                                               # Print blank line
	@echo "===== Running Pthreads (2 threads) ====="       # Print section header
	$(PTHREADS) $(GRID) $(ITERS) $(TOL) 2                  # Run Pthreads with 2 threads
	@echo "===== Running Pthreads (4 threads) ====="       # Print section header
	$(PTHREADS) $(GRID) $(ITERS) $(TOL) 4                  # Run Pthreads with 4 threads
	@echo ""                                               # Print blank line
	@echo "===== Running MPI (2 processes) ====="          # Print section header
	mpirun --oversubscribe -np 2 $(MPI) $(GRID) $(ITERS) $(TOL)   # Run MPI with 2 processes (oversubscribe allows more procs than cores)
	@echo "===== Running MPI (4 processes) ====="          # Print section header
	mpirun --oversubscribe -np 4 $(MPI) $(GRID) $(ITERS) $(TOL)   # Run MPI with 4 processes
	@echo ""                                               # Print blank line
	@echo "===== Running Hybrid (2 procs x 2 threads) =====" # Print section header
	mpirun --oversubscribe -np 2 $(HYBRID) $(GRID) $(ITERS) $(TOL) 2  # Run Hybrid with 2 MPI processes, each using 2 OpenMP threads
	@echo ""                                               # Print blank line
	@echo "All benchmarks complete."                       # Print completion message
