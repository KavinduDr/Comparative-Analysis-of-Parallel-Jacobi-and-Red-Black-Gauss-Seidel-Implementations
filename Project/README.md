# Architectural Scaling of Iterative Linear Solvers

## A Comparative Analysis of Parallel Jacobi and Red-Black Gauss-Seidel Implementations

**Course:** EE7218/EC7207 High Performance Computing — Academic Year 2025/2026

**Group Members:**
- EG/2021/4754 — Rathnayaka I.G.T.A.
- EG/2021/4758 — Rathnayaka R.M.K.D.
- EG/2021/4760 — Rifath M.F.M.

---

## Table of Contents
1. [Problem Description](#problem-description)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Build Instructions](#build-instructions)
5. [Running the Solvers](#running-the-solvers)
6. [Implementation Details](#implementation-details)
7. [Parameters](#parameters)
8. [Expected Output](#expected-output)
9. [Performance Analysis Guide](#performance-analysis-guide)

---

## Problem Description

This project solves the **2D Poisson equation** on a unit square using iterative methods:

```
-∇²u(x,y) = f(x,y)    on [0,1] × [0,1]
u = 0                    on the boundary
```

where `f(x,y) = 2π² sin(πx) sin(πy)`, which has the known exact solution `u(x,y) = sin(πx) sin(πy)`.

The equation is discretized using a **5-point finite difference stencil** on an `n × n` grid, producing a strictly **diagonally dominant** linear system `Ax = b` with `n²` unknowns.

Two iterative solvers are implemented and compared:

1. **Jacobi Method** — Updates all points using only values from the previous iteration. Naturally parallelizable as there are no data dependencies within an iteration.

2. **Red-Black Gauss-Seidel Method** — Uses checkerboard ordering to split grid points into two independent sets (red and black). Updates red points first, then black points using the newly computed red values. Achieves faster convergence than Jacobi while remaining parallelizable within each color phase.

Both solvers are implemented across a **four-tier parallel hierarchy**:

| Tier | Technology | Parallelism Model | Source File |
|------|------------|-------------------|-------------|
| 0 | Serial (C) | Baseline | `src/serial.c` |
| 1 | OpenMP | Shared Memory | `src/openmp_solver.c` |
| 1 | POSIX Threads | Shared Memory | `src/pthreads_solver.c` |
| 2 | MPI | Distributed Memory | `src/mpi_solver.c` |
| 3 | MPI + OpenMP | Hybrid | `src/hybrid_solver.c` |
| 4 | CUDA | GPU Acceleration | `src/cuda_solver.cu` |

---

## Project Structure

```
Project/
├── include/
│   └── common.h              # Shared utilities (grid setup, timing, RMSE)
├── src/
│   ├── serial.c              # Serial Jacobi & Red-Black GS
│   ├── openmp_solver.c       # OpenMP parallel implementation
│   ├── pthreads_solver.c     # POSIX Threads implementation
│   ├── mpi_solver.c          # MPI distributed implementation
│   ├── hybrid_solver.c       # MPI + OpenMP hybrid implementation
│   └── cuda_solver.cu        # CUDA GPU implementation
├── bin/                       # Compiled binaries (created by make)
├── Makefile                   # Build system
└── README.md                  # This file
```

---

## Prerequisites

### Required
- **GCC** (version 4.9+ for OpenMP 4.0 support)
- **MPI** (OpenMPI or MPICH)
  ```bash
  # Ubuntu/Debian
  sudo apt-get install gcc libopenmpi-dev openmpi-bin

  # CentOS/RHEL
  sudo yum install gcc openmpi openmpi-devel

  # macOS (Homebrew)
  brew install gcc open-mpi
  ```

### Optional (for CUDA)
- **NVIDIA CUDA Toolkit** (version 10.0+)
- NVIDIA GPU with compute capability 5.0+
  ```bash
  # Verify CUDA installation
  nvcc --version
  nvidia-smi
  ```

---

## Build Instructions

### Build all CPU versions (recommended start)
```bash
make all_cpu
```

### Build individual targets
```bash
make serial        # Serial baseline
make openmp        # OpenMP version
make pthreads      # POSIX Threads version
make mpi           # MPI version
make hybrid        # Hybrid MPI+OpenMP version
make cuda          # CUDA GPU version (requires NVCC)
```

### Build everything (including CUDA)
```bash
make all
```

### Clean build artifacts
```bash
make clean
```

### macOS Note
If using Homebrew GCC on macOS, update the Makefile compilers:
```makefile
CC = gcc-14       # or your installed version
MPICC = mpicc     # should work if OpenMPI was installed via brew
```

---

## Running the Solvers

All solvers accept command-line arguments:

### 1. Serial
```bash
./bin/serial [grid_size] [max_iterations] [tolerance]

# Examples:
./bin/serial                     # Default: 100x100 grid
./bin/serial 200                 # 200x200 grid
./bin/serial 200 5000 1e-6       # Full options
```

### 2. OpenMP
```bash
./bin/openmp_solver [grid_size] [max_iterations] [tolerance] [num_threads]

# Examples:
./bin/openmp_solver 200 5000 1e-6 2    # 2 threads
./bin/openmp_solver 200 5000 1e-6 4    # 4 threads
./bin/openmp_solver 200 5000 1e-6 8    # 8 threads
```

### 3. POSIX Threads
```bash
./bin/pthreads_solver [grid_size] [max_iterations] [tolerance] [num_threads]

# Examples:
./bin/pthreads_solver 200 5000 1e-6 2
./bin/pthreads_solver 200 5000 1e-6 4
```

### 4. MPI
```bash
mpirun -np <num_processes> ./bin/mpi_solver [grid_size] [max_iterations] [tolerance]

# Examples:
mpirun -np 2 ./bin/mpi_solver 200 5000 1e-6
mpirun -np 4 ./bin/mpi_solver 200 5000 1e-6
mpirun --oversubscribe -np 8 ./bin/mpi_solver 200 5000 1e-6
```

### 5. Hybrid MPI + OpenMP
```bash
mpirun -np <num_processes> ./bin/hybrid_solver [grid_size] [max_iter] [tol] [omp_threads]

# Examples:
mpirun -np 2 ./bin/hybrid_solver 200 5000 1e-6 2    # 2 procs × 2 threads = 4 cores
mpirun -np 2 ./bin/hybrid_solver 200 5000 1e-6 4    # 2 procs × 4 threads = 8 cores
mpirun -np 4 ./bin/hybrid_solver 200 5000 1e-6 2    # 4 procs × 2 threads = 8 cores
```

### 6. CUDA
```bash
./bin/cuda_solver [grid_size] [max_iterations] [tolerance]

# Examples:
./bin/cuda_solver 200 5000 1e-6
./bin/cuda_solver 500 10000 1e-6      # Larger grid benefits more from GPU
```

### Run All Benchmarks Automatically
```bash
make run_all                        # Default: 200x200 grid
make run_all GRID=300 ITERS=8000    # Custom parameters
```

---

## Implementation Details

### Jacobi Method
```
For each iteration:
    For each grid point (i,j):
        u_new(i,j) = (u_old(i-1,j) + u_old(i+1,j) +
                       u_old(i,j-1) + u_old(i,j+1) + h²f(i,j)) / 4
    Check convergence: max|u_new - u_old| < tolerance
```
- Requires two arrays (old and new values)
- All points can be updated independently — ideal for parallelism
- Converges slower than Gauss-Seidel

### Red-Black Gauss-Seidel Method
```
For each iteration:
    Phase 1 (RED):   Update all points where (i+j) is even
    Phase 2 (BLACK): Update all points where (i+j) is odd
    Check convergence: max|u_new - u_old| < tolerance
```
- In-place update (single array)
- Within each phase, all points are independent (neighbors are the other color)
- Converges roughly 2x faster than Jacobi

### Parallelization Strategies

| Implementation | Decomposition | Synchronization |
|---------------|---------------|-----------------|
| **OpenMP** | Loop-level (`#pragma omp parallel for collapse(2)`) | Implicit barriers between parallel regions |
| **Pthreads** | Row-block partitioning | `pthread_barrier` + `pthread_mutex` for reduction |
| **MPI** | Row-wise domain decomposition | `MPI_Sendrecv` (ghost rows) + `MPI_Allreduce` (convergence) |
| **Hybrid** | MPI rows + OpenMP loops within each process | MPI communication + OpenMP barriers |
| **CUDA** | One thread per grid point (2D thread blocks) | Kernel launches act as barriers; GPU reduction for convergence |

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_size` | 100 | Number of interior grid points per dimension (total unknowns = n²) |
| `max_iterations` | 10000 | Maximum number of solver iterations |
| `tolerance` | 1e-6 | Convergence threshold (max absolute change between iterations) |
| `num_threads` | 4 | Number of OpenMP/Pthreads threads |

### Recommended Grid Sizes for Testing

| Grid Size | Unknowns | Recommended For |
|-----------|----------|-----------------|
| 50 | 2,500 | Quick verification |
| 100 | 10,000 | Default testing |
| 200 | 40,000 | Performance comparison |
| 500 | 250,000 | Scalability analysis |
| 1000 | 1,000,000 | Large-scale benchmarking |

---

## Expected Output

Each solver prints a summary like:
```
=============================================================
  Method : Jacobi (Serial)
  Grid   : 200 x 200  (40000 unknowns)
  Iters  : 3847
  Time   : 2.451234 seconds
  Residual (inf-norm) : 9.87e-07
  RMSE vs exact       : 1.23e-04
=============================================================
```

**Key metrics:**
- **Iters**: Number of iterations to converge (Jacobi > Red-Black GS)
- **Time**: Wall-clock execution time in seconds
- **Residual**: Maximum absolute residual of the discrete equation (should be < tolerance)
- **RMSE vs exact**: Root Mean Square Error against the known analytical solution

---

## Performance Analysis Guide

To produce the data required for the analysis report:

### 1. Scaling Study (vary thread/process count)
```bash
# OpenMP thread scaling
for t in 1 2 4 8; do
    echo "--- OpenMP $t threads ---"
    ./bin/openmp_solver 300 10000 1e-6 $t
done

# MPI process scaling
for p in 1 2 4 8; do
    echo "--- MPI $p processes ---"
    mpirun --oversubscribe -np $p ./bin/mpi_solver 300 10000 1e-6
done
```

### 2. Problem Size Scaling (vary grid size)
```bash
for n in 50 100 200 300 500; do
    echo "--- Grid $n x $n ---"
    ./bin/serial $n 10000 1e-6
    ./bin/openmp_solver $n 10000 1e-6 4
    mpirun --oversubscribe -np 4 ./bin/mpi_solver $n 10000 1e-6
done
```

### 3. Accuracy Comparison (RMSE between serial and parallel)
The RMSE between serial and each parallel version should be near machine epsilon (~1e-15) since all implementations use the same algorithm. The RMSE vs exact solution measures the accuracy of the numerical method itself.

### 4. Speedup Calculation
```
Speedup(p) = T_serial / T_parallel(p)
Efficiency(p) = Speedup(p) / p
```

### 5. Convergence Comparison
- Jacobi requires more iterations than Red-Black GS for the same tolerance
- Both methods produce solutions with similar RMSE vs exact solution
- The RMSE between Jacobi and Red-Black GS solutions shows they converge to slightly different discrete solutions (both valid)

---

## Troubleshooting

### MPI errors on single machine
Use `--oversubscribe` flag:
```bash
mpirun --oversubscribe -np 8 ./bin/mpi_solver 200
```

### OpenMP not working
Verify your compiler supports OpenMP:
```bash
gcc -fopenmp -o test_omp -x c - <<< '#include <omp.h>
#include <stdio.h>
int main(){printf("Threads: %d\n",omp_get_max_threads());return 0;}'
./test_omp
```

### CUDA build fails
- Ensure `nvcc` is in your PATH
- Check GPU compute capability and update `-arch=sm_XX` in Makefile
- If no GPU is available, build only CPU targets: `make all_cpu`

### Pthreads barrier not available
On some systems, you may need to add `-D_XOPEN_SOURCE=600` to CFLAGS in the Makefile.

---

## Individual Contributions & Deliverables

### **Rathnayaka I.G.T.A. (EG/2021/4754 — Tharanga Anuradha)**
- **Core OpenMP Parallel Solvers (`openmp_solver.c`)**: Programmed the multi-threaded parallel execution flow for the Jacobi solver (using loop collapse) and Red-Black Gauss-Seidel solvers (segmented red and black multi-threaded phases) with dynamic thread allocation, CLI argument handling, and convergence reduction patterns.
- **POSIX Threads macOS Barrier Shim (`pthreads_solver.c`)**: Designed and successfully implemented a custom POSIX thread barrier (`pthread_barrier_t`) that supports macOS natively, bypassing default Apple OS limitations, alongside worker thread block partitioning.
- **Interactive Python Tkinter/Matplotlib Desktop GUI Application (`experiment_app.py`)**: Built the entire experiment laboratory dashboard to test solvers asynchronously, conduct thread scaling benchmarks, and plot speedup, performance, and complexity growth charts in real time.
- **macOS Adaptation & Compiler Integration**: Configured Makefile compilation suites specifically for macOS Apple Silicon compatibility (targeting Homebrew `gcc-14` and handling OpenMP multi-threading linkages cleanly).
- **HPC Project System Integration**: Merged MPI, CUDA, Hybrid, and Pthreads implementations into the unified codebase, resolving merge conflicts and verifying complete pipeline build accuracy.
