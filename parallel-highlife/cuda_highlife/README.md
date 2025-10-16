# CUDA HighLife

Parallel implementation of HighLife using **CUDA** with 1-D arrays.

**Setup:**
- Pattern: Replicator
- World sizes: 1K x 1K – 65K x 65K
- Block sizes: 8–1024
- Iterations: 1024

**Program arguments:**
The executable accepts four arguments:
./highlife <pattern> <world_size> <iterations> <block_size>

where:
-   `<pattern>` - initial world program (0 -5 as  defined in the assignment)
-   `<world_size>` - dimension `N` of the N x N world
-   `<iterations>` - number of simulation steps
-   `<block_size>` - CUDA thread block size


**Build example:**
```bash
nvcc -O3 -gencode arch=compute_XY,code=sm_XY highlife.cu -o highlife
```
>[!Note] 
>Replace `XY` with your GPU's **compute capability** 

**Run example:**
```bash
./highlife 5 65536 1024 128
```
**Result summary:**
- Fastest block size: **128**
- Peak: **8.65×10¹⁰ cell updates/s**
- ≈900× faster than serial

