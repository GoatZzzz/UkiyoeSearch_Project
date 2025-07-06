from pathlib import Path
import numpy as np
import pandas as pd

# 设置数据集路径
dataset_version = "lite"  # 选择 "lite" 或 "full"
unsplash_dataset_path = Path("/Users/zhu_yangyang/Desktop/Unsplash_Search_Project/unsplash-dataset") / dataset_version
features_path = Path("/Users/zhu_yangyang/Desktop/Unsplash_Search_Project/unsplash-dataset") / dataset_version / "features"

# 读取照片的元数据
photos = pd.read_csv(unsplash_dataset_path / "photos.tsv", sep='\t', header=0)

# 加载照片特征向量和对应的照片 ID
photo_features = np.load(features_path / "features.npy")
photo_ids = pd.read_csv(features_path / "photo_ids.csv")
photo_ids = list(photo_ids['photo_id'])

print(f"Loaded {len(photo_ids)} photos and their features.")