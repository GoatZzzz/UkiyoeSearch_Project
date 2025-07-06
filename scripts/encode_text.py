# scripts/encode_text.py
import torch
from transformers import CLIPTokenizer, CLIPModel

def encode_text(text, clip_model, clip_tokenizer):
    inputs = clip_tokenizer([text], padding=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
        # 这里要看你具体使用的 CLIPModel，如果是 openai/clip-vit-base-patch32
        # 可能使用 clip_model.encode_text(inputs) 或 get_text_features() 不同版本接口
    # 做一下后处理 (归一化等), 看需要
    return text_embeddings[0].cpu().numpy()

def main():
    # 1. 加载CLIP模型
    model_name_or_path = "openai/clip-vit-base-patch32"
    clip_model = CLIPModel.from_pretrained(model_name_or_path)
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_name_or_path)

    # 2. 对输入文本编码
    text = "A beautiful ukiyo-e painting of a mountain"
    text_vec = encode_text(text, clip_model, clip_tokenizer)
    print("文本向量维度:", text_vec.shape)

    # 后续可以把 text_vec 传给 search.py 的函数进行搜索

if __name__ == "__main__":
    main()