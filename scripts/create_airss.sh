#!/bin/bash
####### presetting ###########
numThread=4                                # 进程数量
numStructure=100                           # 结构数量
inputFileName="GaO_diamond_interface.cell" # 输入文件名称
####### end presetting #######
# 定义退出方式 2 = ctrl + c
trap "exec 6>&-;exec 6<&-;wxit 0" 2
# 创建管道
tmp_fifofile=/tmp/$$.fifo
mkfifo $tmp_fifofile
exec 6<>$tmp_fifofile
rm $tmp_fifofile
# 放入令牌
for ((i = 0; i < $numThread; i++)); do
    echo
done >&6
# 执行任务
for i in $(seq $numStructure); do
    read -u6
    {
        ./buildcell <$inputFileName >"tmp."$i.$inputFileName
        echo >&6
    } &
done
# 等待最后的任务
wait
# 退出
exec 6>&-
exec 6<&-
