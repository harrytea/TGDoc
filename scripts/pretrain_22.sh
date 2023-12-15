#!/bin/bash

########### USE THIS FOR BOTH ###########
PROMPT_VERSION=v1
########### DO NOT CHANGE ###########

export NCCL_NET=Socket
# export PATH=/opt/cuda/12.0.1_525.85.12/bin:/opt/cuda/12.0.1_525.85.12/lib64:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
# export PATH=/opt/cuda/11.7.1_515.65.01/bin:/opt/cuda/11.7.1_515.65.01/nvvm:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
export LD_LIBRARY_PATH=/home/sist/wangyh/miniconda3/envs/llava/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/stubs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/targets/x86_64-linux/lib/stubs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/stubs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/targets/x86_64-linux/lib/stubs






# export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
# export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

torchrun --nnodes=1 --nproc_per_node=8 --master_port=25013 \
    llava/train/train_mem.py \
    --model_name_or_path /home/sist/wangyh/LLaVA/checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /home/sist/wangyh/dataset/datasets/instruction \
    --image_folder /home/sist/wangyh/dataset/datasets/images \
    --data_stage pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end \
    --bf16 True \
    --output_dir ./checkpoints/llava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none
