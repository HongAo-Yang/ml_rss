start_tm=$(date +%s%N)

mpirun -np 8 python main2.py

end_tm=$(date +%s%N)
use_tm=$(echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}')
echo total_time $use_tm
