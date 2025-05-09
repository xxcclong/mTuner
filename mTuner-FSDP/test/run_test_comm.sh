#!/bin/sh
for replica_rate in $(seq 0.5 0.1 0.5); do
    CUDA_VISIBLE_DEVICES=0,2,5,6 ~/fthub/bin/python test_comm.py --replica_rate=$replica_rate
done


# for replica_rate in $(seq 0 0.1 1); do
#     CUDA_VISIBLE_DEVICES=0,1,2,3,4 python3 test_comm.py --replica_rate=$replica_rate
# done  
# srun -N 2 -n 4 --gres=gpu:2 run_test_comm.sh


