PROMPT_DATA_PRE = [
    ("LLaVA-CC3M-Pretrain-595K/chat.json", 1.0),
    ("LLaVAR-pretrain/chat_llavar.json", 1.0),  # 422315
    ("PPT_max_pretrain/ppt_ocr_max_in_2048.json", 1.0),  # 98903
]

PROMPT_DATA_TUNE = [
    ("grounding_finetune/finetune_grounding_new.json", 1.0),  # 703
    ("LLaVA-Instruct-150K/llava_instruct_150k.json", 1.0),
    ("LLaVAR-finetune/llavar_instruct_16k.json", 1.0),  # 15786
    ("book/book.json", 1.0),  # 11784
] 