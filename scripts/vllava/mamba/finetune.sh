#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${2:-6}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16688
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
GLOBAL_BATCH_SIZE=240
LOCAL_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=mamba_sft
RUN_NAME=mamba_sft_929
DATA_DIR=/data/vllm/datasets/VideoGPT-plus_Training_Dataset/instruction_tuning
OUTP_DIR=/data/vllm/VideoLLaMA2-main/work_dirs

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    /data/vllm/VideoLLaMA2-main/videollama2/train_flash_attn.py \
    --deepspeed /data/vllm/VideoLLaMA2-main/scripts/zero2.json \
    --lora_enable True --lora_r 256 --lora_alpha 512 --mm_projector_lr 2e-5 \
    --version mistral \
    --vision_tower /data/vllm/ckpt/AI-ModelScope/clip-vit-large-patch14 \
    --mm_projector_type mamba_scan \
    --model_name_or_path /data/vllm/ckpt/AI-ModelScope/Mistral-7B-Instruct-v0___2 \
    --data_path   ${DATA_DIR}/video_sft_638k.json \
    --data_folder ${DATA_DIR}/ \
    --pretrain_mm_mlp_adapter /data/vllm/VideoLLaMA2-main/work_dirs/mamba_pretrain/pretrain_926/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --num_frames 16 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/finetune_${RUN_NAME} \
    --num_train_epochs 3 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 30 \
    --model_max_length 3000 \
    --gradient_checkpointing False \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --run_name $RUN_NAME > mamba_sft_929.log
