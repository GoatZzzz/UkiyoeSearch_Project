import clip
import torch
from PIL import Image

# 加载 CLIP 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 定义函数，计算一批图像的特征向量
def compute_clip_features(photos_batch):
    # 从文件中加载所有图像
    photos = [Image.open(photo_file) for photo_file in photos_batch]
    
    # 对所有图像进行预处理
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # 编码图像并计算特征向量
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)  # 标准化

    # 将特征向量转移到 CPU 并转换为 numpy 数组
    return photos_features.cpu().numpy()