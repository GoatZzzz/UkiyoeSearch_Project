#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征提取脚本 - 使用 ViT-L/14 模型提取768维特征
"""

import math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import clip
from PIL import Image
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_clip_model():
    """加载ViT-L/14 CLIP模型"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading ViT-L/14 CLIP model on device: {device}")
    
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    
    return model, preprocess, device

def compute_clip_features(photos_batch, model, preprocess, device):
    """计算一批图像的CLIP特征向量"""
    try:
        # 加载图像
        photos = []
        for photo_file in photos_batch:
            try:
                image = Image.open(photo_file).convert('RGB')
                photos.append(image)
            except Exception as e:
                logger.error(f"Error loading image {photo_file}: {e}")
                continue
        
        if not photos:
            return None
        
        # 预处理所有图像
        photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)
        
        with torch.no_grad():
            # 编码图像并计算特征向量 (768维)
            photos_features = model.encode_image(photos_preprocessed)
            photos_features /= photos_features.norm(dim=-1, keepdim=True)  # 标准化
        
        return photos_features.cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error computing features: {e}")
        return None

def extract_features_batch_processing():
    """批量处理图片特征提取"""
    # 设置路径
    base_path = Path("/home/zhu01/UkiyoeSearch_Project")
    photos_path = base_path / "ukiyoe_dataset" / "photos"
    features_path = base_path / "ukiyoe_dataset" / "features"
    
    # 创建特征目录
    features_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    photos_files = list(photos_path.glob("*.jpg"))
    if not photos_files:
        logger.error(f"No jpg files found in {photos_path}")
        return
    
    logger.info(f"Found {len(photos_files)} images to process")
    
    # 加载CLIP模型
    model, preprocess, device = load_clip_model()
    
    # 设置批量大小
    batch_size = 16
    batches = math.ceil(len(photos_files) / batch_size)
    
    logger.info(f"Processing {len(photos_files)} images in {batches} batches")
    
    # 清理旧的批次文件
    for old_file in features_path.glob("*.npy"):
        if old_file.name.startswith("0") and old_file.name.endswith(".npy"):
            old_file.unlink()
    for old_file in features_path.glob("*.csv"):
        if old_file.name.startswith("0") and old_file.name.endswith(".csv"):
            old_file.unlink()
    
    # 批量处理每一组照片
    for i in tqdm(range(batches), desc="Processing batches"):
        batch_ids_path = features_path / f"{i:010d}.csv"
        batch_features_path = features_path / f"{i:010d}.npy"
        
        # 获取当前批次的文件
        batch_files = photos_files[i*batch_size : (i+1)*batch_size]
        
        # 计算特征向量
        batch_features = compute_clip_features(batch_files, model, preprocess, device)
        
        if batch_features is not None:
            # 保存特征
            np.save(batch_features_path, batch_features)
            
            # 保存照片ID
            photo_ids = [photo_file.stem for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
            
            logger.info(f"Batch {i+1}/{batches} completed: {len(batch_files)} images, features shape: {batch_features.shape}")
        else:
            logger.error(f"Failed to process batch {i+1}/{batches}")

def merge_features():
    """合并所有批次的特征文件"""
    logger.info("Merging feature files...")
    
    base_path = Path("/home/zhu01/UkiyoeSearch_Project")
    features_path = base_path / "ukiyoe_dataset" / "features"
    
    # 获取所有批次文件
    feature_files = sorted(features_path.glob("*.npy"))
    id_files = sorted(features_path.glob("*.csv"))
    
    # 过滤出批次文件（以数字开头的文件）
    batch_feature_files = [f for f in feature_files if f.name.startswith("0") and f.name != "features.npy"]
    batch_id_files = [f for f in id_files if f.name.startswith("0") and f.name != "photo_ids.csv"]
    
    if not batch_feature_files:
        logger.error("No batch feature files found")
        return
    
    logger.info(f"Found {len(batch_feature_files)} feature files to merge")
    
    # 合并特征
    all_features = []
    all_photo_ids = []
    
    for feature_file, id_file in zip(batch_feature_files, batch_id_files):
        try:
            # 加载特征
            features = np.load(feature_file)
            all_features.append(features)
            
            # 加载ID
            photo_ids = pd.read_csv(id_file)['photo_id'].tolist()
            all_photo_ids.extend(photo_ids)
            
            logger.info(f"Loaded {features.shape[0]} features from {feature_file.name}")
        except Exception as e:
            logger.error(f"Error loading {feature_file}: {e}")
            continue
    
    if not all_features:
        logger.error("No features to merge")
        return
    
    # 合并所有特征
    merged_features = np.vstack(all_features)
    
    # 保存合并后的特征
    merged_features_path = features_path / "features.npy"
    merged_ids_path = features_path / "photo_ids.csv"
    
    np.save(merged_features_path, merged_features)
    pd.DataFrame(all_photo_ids, columns=['photo_id']).to_csv(merged_ids_path, index=False)
    
    logger.info(f"Merged features saved: {merged_features.shape} -> {merged_features_path}")
    logger.info(f"Photo IDs saved: {len(all_photo_ids)} -> {merged_ids_path}")
    
    # 验证特征维度
    if merged_features.shape[1] != 768:
        logger.error(f"Feature dimension mismatch: expected 768, got {merged_features.shape[1]}")
    else:
        logger.info("Feature dimension verified: 768 (ViT-L/14)")

def main():
    """主函数"""
    logger.info("Starting ViT-L/14 feature extraction...")
    
    # 1. 批量提取特征
    extract_features_batch_processing()
    
    # 2. 合并特征文件
    merge_features()
    
    logger.info("Feature extraction completed!")

if __name__ == "__main__":
    main() 