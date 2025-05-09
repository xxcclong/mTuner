model_sizes=(7 13 30 70)
seqs=(1024 2048 4096 8192)


for seq in "${seqs[@]}"; do
    for model_size in "${model_sizes[@]}"; do
        python dp.py --model_size $model_size --seq_len $seq | tee -a dp_output.txt
    done
done

cat dp_output.txt | grep "time cost"