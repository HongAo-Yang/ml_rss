start_tm=$(date +%s%N)

./clean.sh
python3 main2.py 2>err.log

end_tm=$(date +%s%N)
use_tm=$(echo $end_tm $start_tm | awk '{ print ($1 - $2) / 1000000000}')
echo total_time $use_tm
