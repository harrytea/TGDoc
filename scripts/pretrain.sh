#!/bin/bash

deepspeed tgdoc/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /llm-cfs-nj/person/harryyhwang/models/vicuna-7b-v1.5 \
    --data_path /llm-cfs-nj/person/harryyhwang/dataset/instruction \
    --image_folder /llm-cfs-nj/person/harryyhwang/dataset/images \
    --data_stage pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/tgdoc-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --save_strategy "steps" \
    --save_steps 20000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to none
