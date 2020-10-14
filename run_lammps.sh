#!/bin/bash
Ncore=12
lammps_path='lammps' # or 'lmp_mpi'
run_top_dir="LAMMPS_inputs_generated"

start_tm=$(date +%s%N)
cd $run_top_dir
dirs=$(ls -d */)
for dir in $dirs; do
    cd $dir
    echo [LOG] running ${dir}
    mpirun -np $Ncore $lammps_path -in in.minimize.lmp >>../../run_lammps.log
    end_tm=$(date +%s%N)
    use_tm=$(echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}')
    echo "[timer] Total time" $use_tm
    cd ..
done
