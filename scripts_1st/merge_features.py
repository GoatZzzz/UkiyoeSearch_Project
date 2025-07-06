import numpy as np
import pandas as pd
from pathlib import Path

# 设置特征向量保存路径 - 移除 lite 相关代码
features_path = Path("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/features")

# 添加调试信息
print(f"Looking for feature files in: {features_path}")
print(f"Features directory exists: {features_path.exists()}")

# 获取并显示所有特征文件
feature_files = sorted(features_path.glob("*.npy"))
print(f"Found {len(list(feature_files))} .npy files")

# 打印找到的文件名
print("Found following .npy files:")
for f in feature_files:
    print(f"- {f.name}")

# 加载所有 numpy 文件
features_list = [np.load(features_file) for features_file in feature_files]
print(f"Loaded {len(features_list)} feature arrays")

# 将所有特征向量合并为一个 numpy 数组
features = np.concatenate(features_list)
print(f"Concatenated features shape: {features.shape}")
np.save(features_path / "features.npy", features)

# 加载所有照片 ID
id_files = sorted(features_path.glob("*.csv"))
print(f"\nFound {len(list(id_files))} CSV files")
photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in id_files])
print(f"Concatenated {len(photo_ids)} photo IDs")
photo_ids.to_csv(features_path / "photo_ids.csv", index=False)

print("\nMerge completed successfully!")