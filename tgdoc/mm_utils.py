from PIL import Image

import torch
from transformers import StoppingCriteria
from tgdoc.constants import IMAGE_TOKEN_INDEX



def sliding_window(img, stride):
    width, height = img.size
    img_all = img.resize((stride, stride), Image.LANCZOS)

    if width <= stride and height <= stride:
        return [img_all, img_all], [(0, 0), (1, 1)]

    split_images, split_index = [img_all], [(0, 0)]
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            split_images.append(img.crop((j, i, j+stride, i+stride)))
            split_index.append(((i+stride)//stride, (j+stride)//stride))

    return split_images, split_index


def expand2square(pil_img, shard_size, max_grid_num=3):
    width, height = pil_img.size

    max_side = max(width, height)
    scale = min((max_side + shard_size - 1) // shard_size, max_grid_num)
    max_length = shard_size * scale

    if height >= width:
        new_height = max_length
        new_width = max(int(max_length * width / height), 1)
        scale = min((new_width + shard_size - 1) // shard_size, max_grid_num)
        new_width = shard_size * scale
    else:
        new_width = max_length
        new_height = max(int(max_length * height / width), 1)
        scale = min((new_height + shard_size - 1) // shard_size, max_grid_num)
        new_height = shard_size * scale

    resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)  ######## max length
    return resized_img


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    prompt_chunks = [tokenizer(chunk, truncation=True, max_length=4096)['input_ids'] for chunk in prompt.split('<image>')]
    input_ids = prompt_chunks.pop(0)
    for lst in prompt_chunks:
        input_ids.extend([image_token_index] + lst[1:])

    return torch.tensor(input_ids, dtype=torch.long)



# def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
#     prompt_chunks = [tokenizer(chunk, truncation=True, max_length=4096)['input_ids'] for chunk in prompt.split('<image>')]

#     def insert_separator(X, sep):
#         return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

#     input_ids = []
#     offset = 0
#     if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
#         offset = 1
#         input_ids.append(prompt_chunks[0][0])

#     for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
#         input_ids.extend(x[offset:])

#     return torch.tensor(input_ids, dtype=torch.long)


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
