#!/bin/bash
#SBATCH -J pretrain # 作业名称，使用squeue看到的作业名
#SBATCH -N 1 -n 64 # 指定node和核心数量
#SBATCH -o job-%j.log # 把输出结果STDOUT保存在哪一个文件
#SBATCH -e job-%j.err # 把报错结果STDERR保存在哪一个文件
#SBATCH -p GPU-A100-80G # 提交到哪一个分区，方便快速检索合适节点
#SBATCH --qos qos_a100_gpu
#SBATCH --gres=gpu:8 # 需要使用多少GPU，n是需要的数量

echo "SLURM_JOB_PARTITION=$SLURM_JOB_PARTITION"  # 被分配到的队列名称
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"  # 被分配到的节点列表
bash scripts/finetune_20.sh
