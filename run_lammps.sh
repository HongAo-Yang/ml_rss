#!/bin/bash
Ncore=12
lammps_path='lammps' # or 'lmp_mpi'
run_top_dir="LAMMPS_inputs_generated"

cd $run_top_dir
dirs=$(ls -d */)
for dir in $dirs; do
    cd $dir
    echo [LOG] running ${dir}run.sh
    mpirun -np $Ncore $lammps_path -in in.minimize.lmp
    cd ..
done
