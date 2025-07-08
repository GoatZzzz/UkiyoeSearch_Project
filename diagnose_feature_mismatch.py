#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断特征文件是否真的是用ViT-L/14生成的
"""

import numpy as np
import pandas as pd
import torch
import clip
import faiss
from PIL import Image
from pathlib import Path
import random
import os

def test_feature_quality():
    """测试保存的特征质量"""
    print("=" * 60)
    print("测试保存的特征质量")
    print("=" * 60)
    
    # 加载保存的特征
    features = np.load("ukiyoe_dataset/features/features.npy")
    photo_ids = pd.read_csv("ukiyoe_dataset/features/photo_ids.csv")
    
    print(f"特征形状: {features.shape}")
    print(f"特征统计:")
    print(f"  均值: {features.mean():.6f}")
    print(f"  标准差: {features.std():.6f}")
    print(f"  最小值: {features.min():.6f}")
    print(f"  最大值: {features.max():.6f}")
    
    # 检查是否归一化
    norms = np.linalg.norm(features, axis=1)
    print(f"\nL2范数统计:")
    print(f"  均值: {norms.mean():.6f}")
    print(f"  标准差: {norms.std():.6f}")
    print(f"  最小值: {norms.min():.6f}")
    print(f"  最大值: {norms.max():.6f}")
    
    # 检查是否所有特征都是归一化的
    normalized = np.abs(norms - 1.0) < 1e-4
    print(f"归一化特征数量: {normalized.sum()}/{len(features)}")
    
    return features, photo_ids

def regenerate_sample_features():
    """重新生成一些样本特征进行对比"""
    print("\n" + "=" * 60)
    print("重新生成样本特征进行对比")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载ViT-L/14模型
    model, preprocess = clip.load("ViT-L/14", device=device)
    
    # 加载保存的特征
    features = np.load("ukiyoe_dataset/features/features.npy")
    photo_ids = pd.read_csv("ukiyoe_dataset/features/photo_ids.csv")
    
    # 随机选择10张图片
    random.seed(42)
    sample_indices = random.sample(range(len(photo_ids)), 10)
    
    print("重新计算特征对比:")
    matches = 0
    total = 0
    
    for idx in sample_indices:
        photo_id = photo_ids.iloc[idx]['photo_id']
        saved_feature = features[idx]
        
        # 找到图片文件
        photos_dir = Path("ukiyoe_dataset/photos")
        photo_path = None
        for img_file in photos_dir.glob("*.jpg"):
            if photo_id in img_file.name:
                photo_path = img_file
                break
        
        if photo_path and photo_path.exists():
            # 重新计算特征
            try:
                image = Image.open(photo_path).convert('RGB')
                image_input = preprocess(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    new_feature = model.encode_image(image_input)
                    new_feature = new_feature / new_feature.norm(dim=-1, keepdim=True)
                    new_feature = new_feature.cpu().numpy()[0]
                
                # 计算相似度
                cosine_sim = np.dot(saved_feature, new_feature)
                diff = np.abs(saved_feature - new_feature).max()
                
                print(f"  {photo_id[:40]}...")
                print(f"    余弦相似度: {cosine_sim:.6f}")
                print(f"    最大差异: {diff:.6f}")
                
                if cosine_sim > 0.99:
                    matches += 1
                    print("    ✓ 匹配")
                else:
                    print("    ✗ 不匹配")
                
                total += 1
                
            except Exception as e:
                print(f"  {photo_id[:40]}... 处理失败: {e}")
    
    print(f"\n特征匹配度: {matches}/{total} ({matches/total*100:.1f}%)")
    return matches/total if total > 0 else 0

def test_search_distances():
    """测试搜索距离是否正常"""
    print("\n" + "=" * 60)
    print("测试搜索距离")
    print("=" * 60)
    
    # 加载数据
    features = np.load("ukiyoe_dataset/features/features.npy")
    index = faiss.read_index("ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
    
    # 测试自相似性（一个特征向量搜索自己）
    print("测试自相似性搜索:")
    random.seed(42)
    test_indices = random.sample(range(len(features)), 5)
    
    for i, idx in enumerate(test_indices):
        query_vector = features[idx:idx+1].astype('float32')
        distances, indices = index.search(query_vector, 1)
        
        print(f"  测试 {i+1}: 索引 {idx}")
        print(f"    返回索引: {indices[0][0]}")
        print(f"    距离: {distances[0][0]:.8f}")
        
        if indices[0][0] == idx and distances[0][0] < 1e-6:
            print("    ✓ 正常")
        else:
            print("    ✗ 异常")
    
    # 测试典型搜索距离
    print("\n测试典型搜索距离分布:")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-L/14", device=device)
    
    test_queries = ["mountain", "flower", "woman", "man", "water"]
    
    for query in test_queries:
        tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        query_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = index.search(query_vector, 10)
        
        print(f"  '{query}': 距离范围 {distances[0][0]:.3f} - {distances[0][-1]:.3f}")
        
        # 检查距离是否在合理范围内
        if distances[0][0] < 2.0 and distances[0][-1] < 2.0:
            print("    ✓ 距离合理")
        else:
            print("    ✗ 距离可能过大")

def compare_with_fresh_features():
    """与新生成的特征进行对比"""
    print("\n" + "=" * 60)
    print("与新生成的特征进行对比")
    print("=" * 60)
    
    # 如果特征文件是错误的，我们需要重新生成正确的特征
    print("这可能需要重新生成所有特征...")
    
    # 检查是否有原始的特征生成脚本
    possible_scripts = [
        "scripts/extract_features.py",
        "scripts_1st/extract_features.py",
        "extract_features.py"
    ]
    
    for script in possible_scripts:
        if os.path.exists(script):
            print(f"找到特征提取脚本: {script}")
            break
    else:
        print("未找到特征提取脚本，需要创建新的")

def main():
    print("开始诊断特征文件问题...")
    
    # 测试保存的特征质量
    features, photo_ids = test_feature_quality()
    
    # 重新生成样本特征进行对比
    match_rate = regenerate_sample_features()
    
    # 测试搜索距离
    test_search_distances()
    
    # 对比分析
    compare_with_fresh_features()
    
    print("\n" + "=" * 60)
    print("诊断结果")
    print("=" * 60)
    
    if match_rate > 0.8:
        print("✓ 特征文件质量良好，可能是其他问题")
    else:
        print("✗ 特征文件可能有问题，建议重新生成")
        print("\n推荐解决方案:")
        print("1. 重新用ViT-L/14模型提取所有图片特征")
        print("2. 重新构建FAISS索引")
        print("3. 更新Django应用以使用新的索引")

if __name__ == "__main__":
    main() 