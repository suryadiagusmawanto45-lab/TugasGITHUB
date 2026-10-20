#!/bin/bash
#SBATCH --job-name=mc-banjir
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=00:10:00
#SBATCH --output=mc_%j.out

module load python mpi    # sesuaikan dengan modul di cluster
mpirun -n $SLURM_NTASKS python mpi_example.py
