#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-8}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16678
ARG_RANK=0

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=mamba_pretrain
RUN_NAME=829_pre_stage_2_1mamba_2mlp
DATA_DIR=/data/vllm/datasets/VideoGPT-plus_Training_Dataset/pretraining/
OUTP_DIR=/data/vllm/VideoLLaMA2-main/work_dirs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE  \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    /data/vllm/VideoLLaMA2-main/videollama2/train_flash_attn.py \
    --deepspeed /data/vllm/VideoLLaMA2-main/scripts/zero2.json \
    --version plain \
    --vision_tower /data/vllm/ckpt/AI-ModelScope/clip-vit-large-patch14-336 \
    --mm_projector_type mamba_scan \
    --tune_mm_mlp_adapter True \
    --model_name_or_path /data/vllm/ckpt/AI-ModelScope/Mistral-7B-Instruct-v0___2 \
    --data_path   ${DATA_DIR}/pretrain_stage_2_482k_new.json \
    --data_folder ${DATA_DIR} \
    --pretrain_mm_mlp_adapter /data/vllm/VideoLLaMA2-main/work_dirs/mamba_pretrain/pretrain_829_pre_stage_1_1mamba_2mlp/checkpoint-60000/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --num_frames 1 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/pretrain_${RUN_NAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --model_max_length 3000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --run_name $RUN_NAME > 839_pre_stage_two_1mamba_2mlp.log
