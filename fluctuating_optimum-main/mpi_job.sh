#!/bin/bash
#SBATCH --ntasks=100               # number of MPI processes
#SBATCH --mem-per-cpu=4G      # memory; default unit is megabytes
#SBATCH --time=0-00:20           # time (DD-HH:MM)

module load scipy-stack mpi4py

srun python simulate_mpi.py
