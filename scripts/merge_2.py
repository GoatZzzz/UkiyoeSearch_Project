import glob
import re
import pandas as pd

# 只匹配类似 0000000002.csv 的文件名（10位数字）
metadata_files = glob.glob("/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/features/*.csv")

# 严格按 10 位数字匹配（例如 0000001234.csv）
metadata_files = [
    f for f in metadata_files 
    if re.match(r'.*/\d{10}\.csv$', f)  # 路径中包含10位数字的文件名
]

# 按文件名中的数字排序
metadata_files.sort(key=lambda x: int(re.search(r'(\d{10})\.csv$', x).group(1)))

# 合并数据
metadata_df = pd.concat([pd.read_csv(f) for f in metadata_files], ignore_index=True)
print("合并后的元数据条数:", len(metadata_df))  # 预期应为58000