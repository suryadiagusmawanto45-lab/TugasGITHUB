"""Versi MPI (untuk cluster HPC sungguhan). Jalankan: mpirun -n 4 python mpi_example.py
Butuh: pip install mpi4py"""
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
N = 8_000_000
local_n = N // size
rng = np.random.default_rng(1000 + rank)
rain = rng.lognormal(4.4, 0.55, size=(local_n, 7))
local_hits = int((rain.sum(axis=1) > 700.0).sum())
total = comm.reduce(local_hits, op=MPI.SUM, root=0)
if rank == 0:
    print(f"P(banjir) = {total / (local_n * size):.4f} dengan {size} proses")
