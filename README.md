# Parallel HighLife — CUDA, MPI, and Hybrid Implementations

Three progressively parallel implementations of **HighLife**, a cellular automaton similar to Conway’s Game of Life, executed on the AiMOS supercomputer.

| Implementation | Model | Highlights | Report |
|----------------|--------|-------------|---------|
| [`cuda_highlife`](cuda_highlife) | CUDA (Single GPU) | HighLife using 1-D arrays; tested block sizes 8–1024 and world sizes from 1024x1024 up to 65536x65536 | [Report (PDF)](cuda_highlife/cuda_report.pdf) |
| [`mpi_highlife`](mpi_highlife) | MPI (CPU Cluster) | 1-D domain decomposition with ghost-row exchange; tested up to 64 cores | [Report (PDF)](mpi_highlife/mpi_report.pdf) |
| [`hybrid_highlife`](hybrid_highlife) | Hybrid MPI + CUDA (Multi-GPU) | Weak scaling across 1–12 GPUs/cores; tested up to two nodes | [Report (PDF)](hybrid_highlife/hybrid_report.pdf) |
