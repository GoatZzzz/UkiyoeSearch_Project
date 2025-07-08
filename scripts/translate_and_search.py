#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
示例：在命令行下运行:
  python translate_and_search.py
即可输入中文，翻译后用CLIP做文本向量，再用FAISS检索。
"""

import os
import torch
import faiss
import numpy as np
import pandas as pd

from transformers import pipeline

#########################################
# 1) 加载 mBART 翻译模型 (本地)
#########################################
def load_mbart_translator():
    """
    加载本地的 mBART-large-50-many-to-many-mmt 模型，
    并设置中文 -> 英文翻译的参数。
    """
    model_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/models/mbart-large-50-many-to-many-mmt"  # 你的mBART模型本地目录
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    translator = pipeline(
        task="translation",
        model=model_path,
        tokenizer=model_path,
        src_lang="zh_CN",   # 输入语言
        tgt_lang="en_XX",   # 输出语言
        device=device,
        # 如果加载出现报错可以加 use_fast=False 或做其他调整
    )
    return translator

#########################################
# 2) 加载 CLIP (ViT-L/14) - 官方库方式
#########################################
def load_clip_model(device="cpu"):
    """
    使用官方OpenAI的CLIP库：
      pip install git+https://github.com/openai/CLIP.git
    并且加载ViT-L/14型号。
    
    如果你是用Hugging Face的'openai/clip-vit-large-patch14'，
    可以改为Transformers的写法。
    """
    import clip 
    model, preprocess = clip.load("ViT-L/14", device=device)
    return model, preprocess

def encode_text_with_clip(text, clip_model, device="cpu"):
    """
    使用官方CLIP库对英文文本进行Tokenize和Encode，输出(768,)的向量。
    """
    import clip
    tokens = clip.tokenize([text]).to(device)  # batch维度=1
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        # text_features.shape = [1, 768], float32
    return text_features[0].cpu().numpy().astype("float32")

#########################################
# 3) 加载FAISS索引 和 执行检索
#########################################
def load_faiss_index(index_path):
    """加载FAISS索引"""
    return faiss.read_index(index_path)

def faiss_search(query_vector, faiss_index, top_k=5):
    """检索最相似的向量"""
    # 确保是二维数组
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    distances, indices = faiss_index.search(query_vector.astype('float32'), top_k)
    return distances[0], indices[0]

#########################################
# 4) 主流程脚本
#########################################
def main():
    # a) 加载翻译pipeline (mBART)
    translator = load_mbart_translator()

    # b) 加载CLIP
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    clip_model, preprocess = load_clip_model(device)

    # c) 准备中文文本（你也可以用 input() 交互式获取）
    cn_text = "一座美丽的富士山浮世绘"
    print("原始中文:", cn_text)

    # d) 翻译成英文
    # translator(...) 返回一个列表，每个元素是字典 {'translation_text': 'xxx'}
    result = translator(cn_text)
    en_text = result[0]["translation_text"]
    print("翻译结果:", en_text)

    # e) 用CLIP编码英文文本 → (768,)向量
    text_vector = encode_text_with_clip(en_text, clip_model, device)
    print("CLIP文本向量维度:", text_vector.shape)  # 预计(768,)

    # f) 加载FAISS索引 - 使用新的ViT-L/14索引
    index_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index"
    faiss_index = load_faiss_index(index_path)
    print("FAISS索引大小:", faiss_index.ntotal)

    # g) 在索引中搜索
    distances, indices = faiss_search(text_vector, faiss_index, top_k=5)
    print("检索到的Indices:", indices)
    print("对应的Distances:", distances)

    # h) 加载元数据并映射
    metadata_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/merged_metadata.csv"
    metadata_df = pd.read_csv(metadata_path)  # 行数应与faiss_index.ntotal相同(58612)
    print("已加载元数据: 行数 =", len(metadata_df))

    for rank, idx in enumerate(indices):
        row = metadata_df.iloc[idx]
        print(f"\nRank {rank+1}, Index={idx}, Distance={distances[rank]}")
        print("元数据:", row.to_dict())
        # 如果 row 里有 'photo_id' 或 'image_path'，可以用来定位图片

if __name__ == "__main__":
    main()