import os
import torch
import argparse
os.environ['CURL_CA_BUNDLE'] = ''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--TGDoc_model_path", type=str, default="./checkpoints/tgdoc-7b-finetune")
    parser.add_argument("--image_file", default="assets/qua1.jpg", type=str)
    parser.add_argument("--question", default="describe this image", type=str)
    args = parser.parse_args()

    from TGDoc import TGDoc
    # model = LLaVA(model_path=args.LLaVA_model_path, device='cpu', dtype=torch.float32)
    model = TGDoc(model_path=args.TGDoc_model_path, device='cuda', dtype=torch.float16)
    print(model.generate(image=args.image_file, question=args.question))
