This folder contains scripts to run random structure searching progressively.
Temporary structures and trajectories are saved on disk.
## step 1
mpirun -np 4 python generate_initial_random_structures.py
This creates random structures in extxyz format from Ga2O3.cell
## step 2
mpirun -np 4 python relax_structure.py
Minimize the random structures using a GAP potential.
Save the trajectories for next step.
## step 3
mpirun -np 4 python sampling_from_traj.py
Select from RSS trajectories by Boltzmann-weighted flat histogram.
