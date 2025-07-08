# scripts/build_index.py
import os
import faiss
import numpy as np
import pandas as pd

def main():
    # 1. 读取已经合并好的向量
    vectors_path = "/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/features/features.npy"   # 例如: /Users/.../features/vectors.npy
    vectors = np.load(vectors_path).astype('float32')  # shape = (58000, 768) - 使用ViT-L/14的768维特征

    print("加载向量:", vectors.shape)
    
    # 2. 创建IVF索引（示例：IVF_FLAT）
    d = vectors.shape[1]  # 768 (ViT-L/14特征维度)
    nlist = 100           # 可以根据数据量适当调参
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    # 3. 训练索引
    index.train(vectors)
    print("索引已训练")

    # 4. 添加向量
    index.add(vectors)
    print("向量已添加到索引, 索引大小:", index.ntotal)

    # 5. 保存索引
    index_path = "/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index"
    faiss.write_index(index, index_path)
    print(f"索引已保存到: {index_path}")

    # 6. （可选）读取并保存metadata到本地，如有需要也可以在别处统一加载
    #metadata_path = "/home/zhu01/UkiyoeSearch_Project/ukiyoe_dataset/index/metadata.csv"  
    #metadata_df = pd.read_csv(metadata_path)
    # 看你后续怎么使用，这里可能不需要额外保存，因为本身 CSV 就在硬盘。
    # 也可以把metadata_df.to_pickle()保存成pkl，加载更快

if __name__ == "__main__":
    main()
