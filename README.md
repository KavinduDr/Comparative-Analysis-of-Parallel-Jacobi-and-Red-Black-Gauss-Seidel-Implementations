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

- Launch the Interactive Visualization GUI (Python):

  ```bash
  cd Project
  python experiment_app.py
  ```
  *(Note: Requires Python 3, `matplotlib` and `tkinter`. On Windows, the app automatically executes the Linux solver binaries via WSL)*

Notes
- `Makefile` default parameters: `GRID=200`, `ITERS=5000`, `TOL=1e-6` (override with `make run_all GRID=500 ITERS=10000`).
- MPI runs use `mpirun` (examples in `Makefile` use `mpirun --oversubscribe -np N`).
- The `cuda` target requires compatible NVIDIA GPU and `nvcc` configured in PATH.

---

## Contributions & Credits

### **Rathnayaka I.G.T.A. (EG/2021/4754 — Tharanga Anuradha)**
- **Core OpenMP Solver Implementation (`openmp_solver.c`)**: Designed and implemented the primary multi-threaded solver structures for Jacobi (using `#pragma omp parallel for collapse(2)`) and Red-Black Gauss-Seidel color phases (separated red/black parallel updates), incorporating early convergence detection and memory mapping.
- **macOS Pthreads Barrier Support (`pthreads_solver.c`)**: Engineered the custom POSIX-compliant thread barrier shim (`pthread_barrier_t`) enabling smooth barrier synchronization natively on macOS (which lacks standard Pthreads barriers), alongside thread row partitioning and worker architectures.
- **Interactive GUI Desktop Application**: Fully designed and implemented the Tkinter + Matplotlib visualization dashboard (`Project/experiment_app.py`) enabling users to interactively test single solver runs, thread scaling studies (speedup/efficiency plots), and grid scalability analysis.
- **macOS Environment Adaptation**: Customized the project `Makefile` to fully support Homebrew GCC compiler environments (`gcc-14`), resolving OpenMP linking and compilation challenges natively on Apple Silicon.
- **System Integration & Conflict Merging**: Successfully resolved branch conflicts to safely merge high-tier MPI and CUDA parallel solvers with the front-end interface, ensuring a unified cross-platform project structure.

### **Rathnayaka R.M.K.D. (EG/2021/4758)**
- **CUDA Solver Implementation (`cuda_solver.cu`)**: Designed and developed the GPU-accelerated parallel solvers using CUDA. Implemented device kernels for massively parallel grid processing for both Jacobi and Red-Black Gauss-Seidel iterations, significantly reducing computation times for large domains.
- **Hybrid MPI+CUDA Implementation (`hybrid_cuda_solver.cu`)**: Engineered the advanced hybrid parallel solver combining distributed-memory MPI with GPU acceleration. Orchestrated ghost-row communication across MPI processes and localized chunk computation on GPUs to maximize throughput in high-performance cluster environments.
