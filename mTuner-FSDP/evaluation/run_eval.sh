# #!/bin/bash

# # 遍历从0到22的整数
# for i in {0..??}
# do
#    # 将每个整数作为参数传递给eval.py脚本
# #    python eval.py $i
#    CUDA_VISIBLE_DEVICES=0,1,2,3 python eval_gsm8k.py --model /data/siqizhu/llama-2-7b-lora-math --target_model lora_$i --data_file ./data/test/GSM8K_test.jsonl
# #    echo "lora_{$i} done"
#    # 或者如果你的环境中是python3
#    # python3 eval.py $i
# done

#!/bin/bash

# 指定要遍历的目录
TARGET_DIRECTORY="/data/siqizhu/llama-2-7b-lora-math-mar05"

# Iterate over each item in the target directory
echo "Subdirectories in $TARGET_DIRECTORY:"
for item in "$TARGET_DIRECTORY"/*; do
    # Check if the item is a directory
    if [ -d "$item" ]; then
        # Extract the name of the directory
        dirname=$(basename "$item")
        echo "$dirname"
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval_gsm8k.py --model $TARGET_DIRECTORY --target_model $dirname --data_file ./data/test/GSM8K_test.jsonl
    fi
done