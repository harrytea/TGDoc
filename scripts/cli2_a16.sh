# export NCCL_NET=Socket
# # export PATH=/opt/cuda/12.0.1_525.85.12/bin:/opt/cuda/12.0.1_525.85.12/nvvm:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
# # export PATH=/opt/cuda/11.7.1_515.65.01/bin:/opt/cuda/11.7.1_515.65.01/nvvm:/home/sist/wangyh/miniconda3/envs/llava2/bin:$PATH
# export LD_LIBRARY_PATH=/home/sist/wangyh/miniconda3/envs/llava/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64_linux-gnu/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/lib64/stubs
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/12.0.1_525.85.12/targets/x86_64-linux/lib/stubs
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/lib64/stubs
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/11.7.1_515.65.01/targets/x86_64-linux/lib/stubs

export CUDA_VISIBLE_DEVICES="5"
python -m llava.serve.cli \
    --model-path ./checkpoints/llava-7b-finetune \
    --image-file "/data/wangyh/mllms/LLaVA_grounding_lora/watch.jpg" \
    --conv-mode "llava_v1"
    # --query "What are the things I should be cautious about when I visit here?" 