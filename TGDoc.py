import os
import torch

from tgdoc import TGDocLlamaForCausalLM
from tgdoc.conversation import conv_templates
from tgdoc.mm_utils import tokenizer_image_token
from transformers import AutoTokenizer, AutoConfig
import safetensors.torch

from PIL import Image
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IM_START_TOKEN = "<img>"
DEFAULT_IM_END_TOKEN = "</img>"
IMAGE_TOKEN_INDEX = -200
        

class TGDoc:
    def __init__(self, model_path, device, dtype):
        print(f"Loading model from {model_path}")
        config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = TGDocLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, config=config).to(device)

        print("Initializing vision modules")
        model.get_model().initialize_vision_modules(model_args=config)
        vision_tower = model.get_model().get_vision_tower().to(dtype=dtype, device=device)

        print("Initializing adapter modules")
        model.get_model().initialize_adapter_modules()
        model.get_model().mm_projector.to(dtype=dtype, device=device)
        image_processor = vision_tower.image_processor

        print("Loading model weights")
        weights1 = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
        # weights2 = torch.load(os.path.join(model_path, "pytorch_model-00002-of-00002.bin"), map_location='cpu')
        # weights1.update(weights2)
        model.load_state_dict(weights1, strict=False)  # 主要是tokenlizer加载一下

        for n,p in model.named_parameters():
            if p.dtype == torch.float32:
                print(n)

        model.to(dtype=dtype, device=device)
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.dtype = dtype


    def generate(self, image, question, max_new_toekns=256):
        if self.model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question
        conv = conv_templates['v1'].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)  # add
        prompt = conv.get_prompt()
        # print(prompt)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX).unsqueeze(0).to(self.device)


        image = Image.open(image).convert('RGB')
        crop_height = self.image_processor.crop_size['height']
        image = image.resize((crop_height, crop_height), Image.LANCZOS)
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()

        image_tensor = image.to(self.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                do_sample=False,  # 加入随机性
                temperature=0.2,  # 0.2 // 0.9
                max_new_tokens=max_new_toekns
            )
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

        return outputs

