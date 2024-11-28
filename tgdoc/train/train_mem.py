import os
import sys
sys.setrecursionlimit(10000)
sys.path.append('.')
sys.path.append('..')

# Need to call this before importing transformers.
from tgdoc.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

from tgdoc.train.train import train

if __name__ == "__main__":
    train()
