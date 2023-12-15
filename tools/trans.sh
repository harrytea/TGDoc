#!/bin/bash


# # -1的意思是永不退出，直到命令结束
# set timeout -1

attempt_num=0
max_attempts=20

while [ $attempt_num -lt $max_attempts ]
do
  # your task here
  rsync -av --progress -e ssh wangyh@211.86.151.115:/home/sist/wangyh/LLaVA_grounding_1/checkpoint_1_epoch/* /data/wangyh/mllms/LLaVA_grounding_1
  # rsync -av --progress /data/wangyh/mllms/LLaVA_0721_work/checkpoints -e ssh wangyh@211.86.151.115:/home/sist/wangyh/LLaVA_fix_230903/checkpoints/* 
  if [ $? -eq 0 ]; then
    # task success
    break
  else
    # task failed
    attempt_num=$[$attempt_num+1]
    echo "Task failed. Retrying..."
  fi
done


if [ $attempt_num -eq $max_attempts ]; then
  # all retries failed
  echo "Task failed after $max_attempts attempts."
  # exit 1
fi

# rsync 除了支持本地两个目录之间的同步，也支持远程同步。
# 它可以将本地内容，同步到远程服务器。
# $ rsync -av source/ username@remote_host:destination

# 也可以将远程内容同步到本地。
# $ rsync -av username@remote_host:source/ destination
# rsync 默认使用 SSH 进行远程登录和数据传输。

# 由于早期 rsync 不使用 SSH 协议，需要用-e参数指定协议，后来才改的。所以，下面-e ssh可以省略。
# $ rsync -av -e ssh source/ user@remote_host:/destination

# 但是，如果 ssh 命令有附加的参数，则必须使用-e参数指定所要执行的 SSH 命令。
# $ rsync -av -e 'ssh -p 2234' source/ user@remote_host:/destination
# 上面命令中，-e参数指定 SSH 使用2234端口。