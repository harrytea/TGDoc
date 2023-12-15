#!/bin/bash

deepspeed llava/train/train_mem.py \
    --deepspeed /data/wangyh/mllms/LLaVA_grounding_lora/scripts/zero3.json \
    --model_name_or_path /data/wangyh/mllms/LLaVA/checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /data/wangyh/mllms/LLaVA/datasets/instruction \
    --image_folder /data/wangyh/mllms/LLaVA/datasets/images \
    --data_stage finetune \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /data/wangyh/mllms/LLaVA_grounding_cvpr_deep/checkpoints/llava-7b-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
