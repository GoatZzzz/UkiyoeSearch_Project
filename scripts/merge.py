import numpy as np
import glob
import re

# 获取所有.npy文件，并过滤出符合命名规则的分块文件
vector_files = glob.glob("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/features/*.npy")

# 仅保留符合数字命名规则的文件（例如 0000000002.npy）
vector_files = [f for f in vector_files if re.search(r'\d{10}\.npy$', f)]  # 按实际文件名调整正则表达式

# 按文件名中的数字自然排序（例如 0000000002.npy → 2）
vector_files.sort(key=lambda x: int(re.search(r'(\d+)\.npy$', x).group(1)))

# 打印排序后的文件列表（可选，用于调试）
print("排序后的向量文件列表:")
for f in vector_files[:3]:  # 显示前3个文件
    print(f)

# 合并向量
try:
    vectors = np.concatenate([np.load(f) for f in vector_files], axis=0)
    print("合并后的向量维度:", vectors.shape)  # 预期输出 (58000, 512)
except Exception as e:
    print("合并失败:", str(e))