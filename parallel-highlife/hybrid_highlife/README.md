# Hybrid HighLife

Parallel implementation of HighLife using **Hybrid MPI + CUDA** across multiple GPUs and compute nodes.

**Setup:**
- Pattern: Replicator  
- Sub-world per rank: 16K × 16K  
- Iterations: 128  
- Block size: 256  
- GPUs / MPI ranks tested: 1 – 12 (up to 2 nodes on AiMOS)  

Each MPI rank runs on one GPU and handles its own 16K×16K sub-world.  
Ranks exchange ghost rows using `MPI_Isend` / `MPI_Irecv`, synchronize with `MPI_Waitall`,  
and update their sub-world with the CUDA HighLife kernel.

**Program arguments:**  
The executable accepts five arguments:
```bash
./highlife_exe <pattern> <world_size> <iterations> <block_size> <output_flag>
```
where:  
- `<pattern>` – initial world pattern (0–5 as defined in the assignment)  
- `<world_size>` – dimension `N` of each rank’s N×N sub-world  
- `<iterations>` – number of simulation steps  
- `<block_size>` – CUDA thread block size  
- `<output_flag>` – `true` or `false` to print the world (for small tests only)  

**Build example:**
```bash
mpicc -O3 highlife_mpi.c -c -o highlife_mpi.o
nvcc -O3 -gencode arch=compute_XY,code=sm_XY highlife_cuda.cu -c -o highlife_cuda.o
mpicc highlife_mpi.o highlife_cuda.o -o highlife_exe \
      -L/usr/local/cuda/lib64 -lcudadevrt -lcudart -lstdc++
```

> [!IMPORTANT]  
> Replace `XY` with your GPU’s **compute capability**  

**Run example:**
```bash
mpirun -np 12 ./highlife_exe 5 16384 128 256 false
```

**Result summary:**
- Maximum speedup: **≈ 9×** (at 12 GPUs / 2 nodes)  
- Best weak-scaling efficiency: **0.74**  
- Slight efficiency drop with more ranks due to communication overhead,  
  but overall strong scaling and good GPU utilization.  

> [!NOTE]  
> For detailed performance plots and timing data, see [hybrid_report.pdf](hybrid_report.pdf).
