import math
import numpy as np
import pandas as pd
from pathlib import Path
from load_clip_model import compute_clip_features  # 导入计算特征向量的函数

# 设置图片路径 - 移除 lite 层级
photos_path = Path("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/photos")
photos_files = list(photos_path.glob("*.jpg"))

# 添加调试信息
print(f"Photos directory exists: {photos_path.exists()}")
print(f"Found {len(photos_files)} jpg files in photos directory")

# 设置批量大小
batch_size = 16

# 设置保存特征向量的路径 - 移除 lite 层级
features_path = Path("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/features")
features_path.mkdir(parents=True, exist_ok=True)

print(f"Features directory exists: {features_path.exists()}")

# 计算需要处理的批次数量
batches = math.ceil(len(photos_files) / batch_size)
print(f"Will process {len(photos_files)} images in {batches} batches")

# 批量处理每一组照片
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"
    
    # 如果这个批次已经处理过，则跳过
    if not batch_features_path.exists():
        try:
            batch_files = photos_files[i*batch_size : (i+1)*batch_size]

            # 计算特征向量并保存
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)

            # 保存照片 ID 到 CSV 文件
            photo_ids = [photo_file.stem for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
        except Exception as e:
            print(f'Problem with batch {i}: {e}')