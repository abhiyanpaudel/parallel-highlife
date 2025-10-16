# MPI HighLife

Parallel implementation of HighLife using **MPI** with 1-D arrays.

**Setup:**
- Pattern: Replicator  
- Square world size: 16K x 16K  
- Iterations: 32  
- MPI ranks tested: 1, 2, 4, 8, 16, 32, 64  

**Program arguments:**
The executable accepts four arguments:
```bash
./highlife_mpi <pattern> <world_size> <iterations> <output_flag>
```

where:
- `<pattern>` – initial world pattern (0–5 as defined in the assignment)
- `<world_size>` – dimension N of the N x N world
- `<iterations>` – number of simulation steps
- `<output_flag>` – true or false to print the world (for small test runs only)

**Build example:**
```bash
mpicc -O3 highlife_mpi.c -o highlife_mpi
```
**Run example**
```bash
mpirun -np 64 ./highlife_mpi 5 16384 32 false
```

**Result summary**
- Maximum speedup: ≈7× (at 64 MPI ranks)
- Highest cell update rate: 1.56×10⁹ updates/s
- Lower performance at 2–4 ranks due to communication overhead


> [!NOTE]
> For detailed performance plots and timing data, see [mpi_report.pdf](mpi_report.pdf).
