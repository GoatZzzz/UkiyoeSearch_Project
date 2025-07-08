#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重新用ViT-B/32提取特征
"""

import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path
import faiss
from tqdm import tqdm
import time

def extract_vit_b32_features():
    """重新提取ViT-B/32特征"""
    print("=" * 60)
    print("重新提取ViT-B/32特征")
    print("=" * 60)
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    print("ViT-B/32模型加载完成")
    
    # 获取图片列表
    photos_dir = Path("ukiyoe_dataset/photos")
    image_files = list(photos_dir.glob("*.jpg"))
    print(f"找到 {len(image_files)} 张图片")
    
    # 提取特征
    features = []
    photo_ids = []
    batch_size = 32
    
    print("开始提取特征...")
    
    for i in tqdm(range(0, len(image_files), batch_size), desc="提取特征"):
        batch_files = image_files[i:i+batch_size]
        batch_images = []
        batch_ids = []
        
        # 加载批次图片
        for img_file in batch_files:
            try:
                image = Image.open(img_file).convert('RGB')
                image_input = preprocess(image)
                batch_images.append(image_input)
                batch_ids.append(img_file.name)
            except Exception as e:
                print(f"处理图片 {img_file} 失败: {e}")
                continue
        
        if batch_images:
            # 批次处理
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                batch_features = model.encode_image(batch_tensor)
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                batch_features = batch_features.cpu().numpy()
            
            features.extend(batch_features)
            photo_ids.extend(batch_ids)
    
    features = np.array(features)
    print(f"特征提取完成，形状: {features.shape}")
    
    # 保存特征
    features_dir = Path("ukiyoe_dataset/features")
    features_dir.mkdir(exist_ok=True)
    
    # 备份旧特征
    old_features_path = "ukiyoe_dataset/features/features.npy"
    if os.path.exists(old_features_path):
        backup_path = f"ukiyoe_dataset/features/features_vit_l14_backup_{int(time.time())}.npy"
        os.rename(old_features_path, backup_path)
        print(f"旧特征文件已备份到: {backup_path}")
    
    # 保存新特征
    np.save("ukiyoe_dataset/features/features.npy", features)
    print("新特征文件已保存到: ukiyoe_dataset/features/features.npy")
    
    # 保存图片ID
    photo_ids_df = pd.DataFrame({'photo_id': photo_ids})
    old_photo_ids_path = "ukiyoe_dataset/features/photo_ids.csv"
    if os.path.exists(old_photo_ids_path):
        backup_path = f"ukiyoe_dataset/features/photo_ids_backup_{int(time.time())}.csv"
        os.rename(old_photo_ids_path, backup_path)
        print(f"旧图片ID文件已备份到: {backup_path}")
    
    photo_ids_df.to_csv("ukiyoe_dataset/features/photo_ids.csv", index=False)
    print("新图片ID文件已保存到: ukiyoe_dataset/features/photo_ids.csv")
    
    return features, photo_ids

def rebuild_faiss_index(features):
    """重建FAISS索引"""
    print("\n" + "=" * 60)
    print("重建FAISS索引")
    print("=" * 60)
    
    # 创建新索引
    d = features.shape[1]  # 512维
    print(f"特征维度: {d}")
    
    # 使用IVF索引
    nlist = min(4096, features.shape[0] // 39)  # 聚类数量
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print(f"创建IVF索引，聚类数量: {nlist}")
    
    # 训练索引
    print("训练索引...")
    index.train(features.astype('float32'))
    
    # 添加向量
    print("添加向量...")
    index.add(features.astype('float32'))
    
    # 设置搜索参数
    index.nprobe = min(32, nlist)
    
    print(f"索引构建完成，包含 {index.ntotal} 个向量")
    
    # 备份旧索引
    old_index_path = "ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index"
    if os.path.exists(old_index_path):
        backup_path = f"ukiyoe_dataset/index/faiss_index_ivf_vit_l14_backup_{int(time.time())}.index"
        os.rename(old_index_path, backup_path)
        print(f"旧索引已备份到: {backup_path}")
    
    # 保存新索引
    faiss.write_index(index, "ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
    print("新索引已保存到: ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
    
    return index

def update_django_app():
    """更新Django应用配置"""
    print("\n" + "=" * 60)
    print("更新Django应用配置")
    print("=" * 60)
    
    # 更新views.py以使用ViT-B/32
    views_content = '''import json
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import logging
import sys
import numpy as np
import pandas as pd
import faiss
import torch
import clip

# 设置日志
logger = logging.getLogger(__name__)

# 全局变量存储模型和数据
model = None
index = None
features = None
photo_ids = None

def load_model_and_index():
    """加载ViT-B/32模型和索引"""
    global model, index, features, photo_ids
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 加载模型
    if model is None:
        model, _ = clip.load("ViT-B/32", device=device)
        logger.info("ViT-B/32 模型加载完成")
    
    # 加载索引
    if index is None:
        index = faiss.read_index("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
        logger.info("FAISS索引加载完成")
    
    # 加载特征和图片ID
    if features is None:
        features = np.load("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/features/features.npy")
        logger.info("特征文件加载完成")
    
    if photo_ids is None:
        photo_ids = pd.read_csv("/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/features/photo_ids.csv")
        logger.info("图片ID文件加载完成")

@csrf_exempt
@require_http_methods(["POST"])
def search_api(request):
    """搜索API - 使用ViT-B/32"""
    try:
        # 加载模型和索引
        load_model_and_index()
        
        # 解析请求
        data = json.loads(request.body)
        query = data.get('query', '')
        limit = int(data.get('limit', 20))
        
        logger.info(f"搜索查询: {query}, 限制: {limit}")
        
        if not query:
            return JsonResponse({'error': '查询不能为空'}, status=400)
        
        # 编码查询文本
        device = next(model.parameters()).device
        tokens = clip.tokenize([query]).to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 搜索
        query_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = index.search(query_vector, limit)
        
        # 构建结果
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            photo_id = photo_ids.iloc[idx]['photo_id']
            
            # 构建图片URL
            if photo_id.endswith('.jpg'):
                image_url = f"/media/{photo_id}"
            else:
                image_url = f"/media/{photo_id}.jpg"
            
            results.append({
                'id': int(idx),
                'photo_id': photo_id,
                'image_url': image_url,
                'distance': float(distance),
                'rank': i + 1
            })
        
        return JsonResponse({
            'results': results,
            'query': query,
            'model_type': 'vit_b32',
            'total_results': len(results),
            'search_time': 'N/A'
        })
        
    except Exception as e:
        logger.error(f"搜索错误: {e}")
        return JsonResponse({'error': str(e)}, status=500)'''
    
    with open("myproject/search_app/views.py", "w", encoding="utf-8") as f:
        f.write(views_content)
    
    print("✓ Django views.py 已更新为使用ViT-B/32")

def test_search_quality():
    """测试搜索质量"""
    print("\n" + "=" * 60)
    print("测试搜索质量")
    print("=" * 60)
    
    # 加载模型和数据
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    features = np.load("ukiyoe_dataset/features/features.npy")
    photo_ids = pd.read_csv("ukiyoe_dataset/features/photo_ids.csv")
    index = faiss.read_index("ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index")
    
    # 测试查询
    test_queries = [
        "Mount Fuji",
        "富士山",
        "cherry blossom",
        "樱花",
        "samurai",
        "武士"
    ]
    
    for query in test_queries:
        print(f"\n查询: '{query}'")
        
        # 编码查询
        tokens = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 搜索
        query_vector = text_features.cpu().numpy().astype('float32')
        distances, indices = index.search(query_vector, 5)
        
        print("搜索结果:")
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            photo_id = photo_ids.iloc[idx]['photo_id']
            print(f"  {i+1}. {photo_id[:50]}... (距离: {dist:.4f})")

def main():
    """主函数"""
    print("开始回到ViT-B/32系统...")
    print("这将提供更好的浮世绘搜索体验")
    
    try:
        # 1. 重新提取ViT-B/32特征
        features, photo_ids = extract_vit_b32_features()
        
        # 2. 重建FAISS索引
        index = rebuild_faiss_index(features)
        
        # 3. 更新Django应用
        update_django_app()
        
        # 4. 测试搜索质量
        test_search_quality()
        
        print("\n" + "=" * 60)
        print("迁移完成！")
        print("=" * 60)
        print("系统现在使用ViT-B/32模型，应该提供更好的搜索结果。")
        print("请重启Django服务器以应用更改：")
        print("cd myproject && python manage.py runserver")
        
    except Exception as e:
        print(f"迁移过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 