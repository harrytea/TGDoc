import copy
import json
from dataclasses import dataclass
import os.path as osp
import torch
import transformers
import random

from tgdoc.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tgdoc.config.data_config import PROMPT_DATA_PRE, PROMPT_DATA_TUNE
from tgdoc import conversation as conversation_lib
from tgdoc.mm_utils import tokenizer_image_token, sliding_window, expand2square
from torch.utils.data import Dataset
from PIL import Image

from torch.distributed import get_rank

def rank0_print(*args):
    if get_rank() == 0:
        print(*args)


# xxx <image>\n --> <im_start><image><im_end>\n xxx
def preprocess_multimodal(sources, data_args):
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = f"{DEFAULT_IMAGE_TOKEN}\n{sentence['value']}".strip()

            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = f"{DEFAULT_IM_START_TOKEN}{replace_token}{DEFAULT_IM_END_TOKEN}"
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_v1(sources, tokenizer):
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    import pdb
    # pdb.set_trace()
    conversations = []
    for source in sources:
        conv.messages = []
        for sentence in source:
            conv.append_message(roles[sentence["from"]], sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer) for prompt in conversations], dim=0)
    targets = input_ids.clone()

    sep = conv.sep + conv.roles[1] + ": "  # Mask targets  ### ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        total_len = target.numel()  # add this
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        for i, round in enumerate(rounds):
            parts = round.split(sep)  # 划分问题和答案
            if round == "" or len(parts) != 2:
                break

            parts[0] += sep  # question + " Assistant: "
            # round_len = len(tokenizer_image_token(round, tokenizer))  # may be wrong in new version hf
            round_len = len(tokenizer_image_token(conv.sep2+round, tokenizer)) - 1 # 去掉多轮对话开头的</s>
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            target[cur_len:cur_len+instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}.  (ignored)")
            print(sources)
    return dict(input_ids=input_ids, labels=targets,)



class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path, tokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = []
        # if data_args.data_stage == "pretrain":
        #     for path, sample_rate in PROMPT_DATA_PRE:
        #         prompt_path = osp.join(data_path, path)
        #         data = json.load(open(prompt_path, "r"))
                
        #         # Sample data according to sample rate
        #         original_size = len(data)
        #         if sample_rate < 1.0:
        #             sample_size = int(original_size * sample_rate)
        #             data = random.sample(data, sample_size)
                    
        #         list_data_dict += data
        #         rank0_print(f"Dataset: {path}, Original size: {original_size}, Sample rate: {sample_rate}, 
        #         Final size: {len(data)}")
        #     rank0_print("Pretrain stage: ", PROMPT_DATA_PRE)
        # else:
        #     for path, sample_rate in PROMPT_DATA_TUNE:
        #         prompt_path = osp.join(data_path, path)
        #         data = json.load(open(prompt_path, "r"))
                
        #         # Sample data according to sample rate
        #         original_size = len(data)
        #         if sample_rate < 1.0:
        #             sample_size = int(original_size * sample_rate)
        #             data = random.sample(data, sample_size)
                    
        #         list_data_dict += data
        #         rank0_print(f"Dataset: {path}, Original size: {original_size}, Sample rate: {sample_rate}, 
        #         Final size: {len(data)}")
        #     rank0_print("Finetuning stage: ", PROMPT_DATA_TUNE)

        # 选择数据集配置
        data_config = PROMPT_DATA_PRE if data_args.data_stage == "pretrain" else PROMPT_DATA_TUNE
        stage_name = "Pretrain" if data_args.data_stage == "pretrain" else "Finetuning"
        
        # 处理每个数据集
        for path, sample_rate in data_config:
            prompt_path = osp.join(data_path, path)
            data = json.load(open(prompt_path, "r"))
            
            original_size = len(data)
            if sample_rate < 1.0:
                data = random.sample(data, int(original_size * sample_rate))
                
            list_data_dict += data
            rank0_print(f"Dataset: {path}, Original size: {original_size}, "
                       f"Sample rate: {sample_rate}, Final size: {len(data)}")
        
        rank0_print(f"{stage_name} stage: ", data_config)
        rank0_print(f"Total dataset size: {len(list_data_dict)}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        input_conv = self.list_data_dict[i]
        # print(input_conv['image_folder'], input_conv['image'])

        # process image and conversation
        data_folder = input_conv['image_folder']
        image_file = input_conv['image']
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        image = Image.open(osp.join(image_folder, data_folder, image_file)).convert('RGB')

        # resize the image to square
        crop_height = processor.crop_size['height']  
        image = image.resize((crop_height, crop_height), Image.LANCZOS)
        image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        # process conversation
        sources = preprocess_multimodal(copy.deepcopy([input_conv["conversations"]]), self.data_args)
        data_dict = preprocess_v1(sources, self.tokenizer)
        data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        data_dict['image'] = image
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """pad the sequence in the same length"""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        pad_token_id = self.tokenizer.pad_token_id
        max_length = self.tokenizer.model_max_length
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)[:, :max_length]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)[:, :max_length]

        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": input_ids.ne(pad_token_id),}

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        return batch


def make_supervised_data_module(tokenizer, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
