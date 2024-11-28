import os
import torch
import transformers

from tgdoc.train.tgdoc_trainer import TGDocTrainer
from tgdoc.config.train_config import ModelArguments, DataArguments, TrainingArguments

from tgdoc.model import *
from tgdoc.utils import *
from tgdoc.data import *
import safetensors.torch
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    _, _, args = parser.parse_args_into_dataclasses()
    local_rank = args.local_rank  # 获取当前rank

    model = TGDocLlamaForCausalLM.from_pretrained(args.model_name_or_path)
    model.config.use_cache = False

    # gradient checkpointing
    if args.gradient_checkpointing:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    # LoRA
    if args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if args.bits == 16:
            if args.bf16:
                model.to(torch.bfloat16)
            if args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)


    # tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    model.get_model().initialize_vision_modules(model_args=args)
    vision_tower = model.get_model().get_vision_tower().to(dtype=torch.bfloat16, device=args.device)

    model.get_model().update_config(args)
    model.get_model().initialize_adapter_modules()
    model.get_model().mm_projector.to(dtype=torch.bfloat16, device=args.device)

    args.image_processor = vision_tower.image_processor

    model.config.tune_mm_mlp_adapter = args.tune_mm_mlp_adapter
    model.config.mm_use_im_start_end = args.mm_use_im_start_end
    if args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
        model.initialize_vision_tokenizer(args, tokenizer=tokenizer)

    if args.data_stage == "finetune":
        weights1 = torch.load(os.path.join(args.model_name_or_path, "pytorch_model-00001-of-00002.bin"), map_location='cpu')
        weights2 = torch.load(os.path.join(args.model_name_or_path, "pytorch_model-00002-of-00002.bin"), map_location='cpu')
        weights1.update(weights2)
        model.load_dict(weights1)  #TODO 用zero3会报错
        for p in model.parameters():
            p.requires_grad = True

    model.to(dtype=torch.bfloat16, device=args.device)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=args)
    trainer = TGDocTrainer(model=model, tokenizer=tokenizer, args=args, **data_module)  # Trainer
    trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=args.output_dir)

if __name__ == "__main__":
    train()
