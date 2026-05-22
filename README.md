# Comparative Analysis of Parallel Jacobi and Red-Black Gauss-Seidel

High Performance Computing (HPC) coursework project comparing parallel implementations
of the Jacobi and Red-Black Gauss-Seidel solvers across several parallel programming
models (Serial, OpenMP, Pthreads, MPI, Hybrid MPI+OpenMP, CUDA).

**Repository**: the main implementation and build artifacts are in the `Project` folder.

**Quick Links**
- **Makefile**: [Project/Makefile](Project/Makefile#L1-L120)
- **Serial source**: [Project/src/serial.c](Project/src/serial.c#L1)
- **CUDA source**: [Project/src/cuda_solver.cu](Project/src/cuda_solver.cu#L1)

**Project layout**
- `Project/` : build scripts, `Makefile`, helper run scripts and `bin/` output
- `Project/src/` : source implementations (`serial.c`, `openmp_solver.c`, `pthreads_solver.c`, `mpi_solver.c`, `hybrid_solver.c`, `cuda_solver.cu`)
- `Project/include/` : common headers (e.g., `common.h`)
- `Project/bin/` : compiled binaries produced by `make`

Prerequisites
- A C compiler (GCC) and `make`.
- OpenMPI (for `mpicc` / `mpirun`) to build/run MPI targets.
- NVIDIA CUDA toolkit (`nvcc`) to build the CUDA target (optional).

Build
1. Change to the `Project` directory:

	`cd Project`

2. Build CPU-only implementations:

	`make all_cpu`

3. Build everything including CUDA (if CUDA toolkit is available):

	`make all`

4. Build a single target, for example CUDA or serial:

	`make cuda`
	`make serial`

Run
- Run all benchmarks (builds CPU targets first):

  `make run_all`

- Run the provided run scripts (bash):

  `./run_benches.sh` or `./run_benches2.sh`

- Run a single binary with parameters `GRID ITERS TOL`, for example:

  `./bin/serial 200 5000 1e-6`

Notes
- `Makefile` default parameters: `GRID=200`, `ITERS=5000`, `TOL=1e-6` (override with `make run_all GRID=500 ITERS=10000`).
- MPI runs use `mpirun` (examples in `Makefile` use `mpirun --oversubscribe -np N`).
- The `cuda` target requires compatible NVIDIA GPU and `nvcc` configured in PATH.

