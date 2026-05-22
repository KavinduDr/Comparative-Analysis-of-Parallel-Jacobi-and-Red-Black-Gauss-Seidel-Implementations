#!/bin/bash
echo "--- Grid Size Scaling ---" > results.txt
for n in 100 200 300; do
    echo "Grid: $n" >> results.txt
    ./bin/serial $n 20000 1e-6 >> results.txt
done
echo "--- Thread Scaling OpenMP (Grid 200) ---" >> results.txt
for t in 1 2 4 8; do
    echo "Threads: $t" >> results.txt
    ./bin/openmp_solver 200 20000 1e-6 $t >> results.txt
done
echo "--- Thread Scaling Pthreads (Grid 200) ---" >> results.txt
for t in 1 2 4 8; do
    echo "Threads: $t" >> results.txt
    ./bin/pthreads_solver 200 20000 1e-6 $t >> results.txt
done
