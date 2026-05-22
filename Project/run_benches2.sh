#!/bin/bash
echo "--- Thread Scaling OpenMP (Grid 200) ---" > results2.txt
for t in 1 2 4 8; do
    echo "Threads: $t" >> results2.txt
    ./bin/openmp_solver 200 5000 1e-6 $t >> results2.txt
done
echo "--- Thread Scaling Pthreads (Grid 200) ---" >> results2.txt
for t in 1 2 4 8; do
    echo "Threads: $t" >> results2.txt
    ./bin/pthreads_solver 200 5000 1e-6 $t >> results2.txt
done
