import torch
from transformers import pipeline

model_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/models/mbart-large-50-many-to-many-mmt"
device = "mps" if torch.backends.mps.is_available() else "cpu"

translator = pipeline(
    task="translation",
    model=model_path,
    tokenizer=model_path,
    src_lang="zh_CN",
    tgt_lang="en_XX",
    device=device
)

cn_text = "一座美丽的富士山浮世绘"
print("原始中文:", cn_text)
result = translator(cn_text)
print("翻译结果:", result[0]["translation_text"])