#!/bin/bash

#SBATCH --job-name=fthub
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=debug
#SBATCH --exclusive


# Runs the "175B" parameter model

mkdir -p output/${SLURM_JOB_ID}
OUTPUT_DIR="${SLURM_JOB_ID}"
OUTPUT_FILE="output/${OUTPUT_DIR}/${SLURM_PROCID}.txt"
# exec > "$OUTPUT_FILE" 2>&1

source ~/.venv/fthub/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1

echo $(pwd)

GPUS_PER_NODE=8
# Change for multinode config
# MASTER_ADDR=localhost
MASTER_ADDR=$(scontrol show JobId=$SLURM_JOB_ID | grep BatchHost | tr '=' ' ' | awk '{print $2}')
MASTER_PORT=12215
NUM_NODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH="logs/$(date "+%m-%d-%H-%M")" 
VOCAB_FILE="/public/home/hkz/repo/hkz/data//vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="/public/home/hkz/repo/hkz/data//merges.txt" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="/public/home/hkz/repo/hkz/data/zbdata/zb_sample_dataset/dataset/c4_text_document" #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

echo $(hostname) $NUM_NODES $NODE_RANK $MASTER_ADDR $MASTER_PORT $WORLD_SIZE

if   [ ${MODEL_SIZE} == 7 ];   then HIDDEN_SIZE=4096;  NUM_HEAD=32; NUM_QUERY_GROUP=32; NUM_LAYERS=32; FFN_HIDDEN_SIZE=11008; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 13 ];  then HIDDEN_SIZE=5120;  NUM_HEAD=40; NUM_QUERY_GROUP=40; NUM_LAYERS=40; FFN_HIDDEN_SIZE=13824; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 30 ];  then HIDDEN_SIZE=6656;  NUM_HEAD=64; NUM_QUERY_GROUP=64;  NUM_LAYERS=60; FFN_HIDDEN_SIZE=17920; NORM_EPS=1e-5;
elif [ ${MODEL_SIZE} == 70 ];  then HIDDEN_SIZE=8192;  NUM_HEAD=64; NUM_QUERY_GROUP=8;  NUM_LAYERS=80; FFN_HIDDEN_SIZE=28672; NORM_EPS=1e-5;
else echo "invalid MODEL_SIZE: ${MODEL_SIZE}"; exit 1
fi

echo "MODEL_SIZE: ${MODEL_SIZE} HIDDEN_SIZE: ${HIDDEN_SIZE} NUM_HEAD: ${NUM_HEAD} NUM_QUERY_GROUP: ${NUM_QUERY_GROUP} NUM_LAYERS: ${NUM_LAYERS} FFN_HIDDEN_SIZE: ${FFN_HIDDEN_SIZE} NORM_EPS: ${NORM_EPS}"

MTUNER="${MTUNER:-1}"

GPT_MODEL_ARGS=(
    --mtuner $MTUNER
    --num-layers $NUM_LAYERS
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_HEAD
    # --group-query-attention 
    --num-query-groups $NUM_QUERY_GROUP
    --seq-length $SEQ
    --max-position-embeddings $SEQ
    --ffn-hidden-size $FFN_HIDDEN_SIZE
    --use-flash-attn
    # --disable-tp-comm-overlap-ag
    # --disable-tp-comm-overlap-rs
    # --tp-comm-overlap
    --bf16
    # --recompute-granularity full
    # --recompute-method uniform
    # --recompute-num-layers 1
)

if [ "$DISABLE_TP_COMM_OVERLAP" != "true" ]; then
    GPT_MODEL_ARGS+=("--tp-comm-overlap")
fi

BS=$BS
TP=$TP
DNUM=8
N="${N:-1}"
GBS=$(( (BS * DNUM * N) / TP ))
echo "global batch size $GBS"


TRAINING_ARGS=(
    --micro-batch-size $BS
    --global-batch-size $GBS
    --train-iters 5
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction 0
    --lr-decay-iters 430000 
)
    # --rampup-batch-size 16 16 5859375 

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size $TP
	--pipeline-model-parallel-size 1
    --sequence-parallel
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --split 949,50,1
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10000 
    --eval-interval 1000 
    # --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

OTHER_ARGS=(
    --transformer-impl transformer_engine
    # --num-layers-per-virtual-pipeline-stage 1
    --timing-log-level 2
    # --profile
    --profile-step-start 3
    --profile-step-end 4
    --use-pytorch-profiler
    --profile-ranks 0 1 2 3 4 5 6 7
    --no-overlap-p2p-communication
    # --overlap-p2p-communication-warmup-flush
)

echo ${GPT_MODEL_ARGS[@]} 
echo ${TRAINING_ARGS[@]}
echo ${MODEL_PARALLEL_ARGS[@]}
echo ${DATA_ARGS[@]}
echo ${EVAL_AND_LOGGING_ARGS[@]}
echo ${OTHER_ARGS[@]}
echo ${DISTRIBUTED_ARGS[@]}

torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${OTHER_ARGS[@]} | tee $OUTPUT_FILE