# scripts/search.py
import faiss
import numpy as np
import pandas as pd

def load_faiss_index(index_path):
    """加载FAISS索引文件"""
    index = faiss.read_index(index_path)
    return index

def load_metadata(metadata_path):
    """加载元数据"""
    metadata_df = pd.read_csv(metadata_path)
    return metadata_df

def search_vectors(query_vector, index, top_k=5):
    """在FAISS索引中检索相似向量, 返回 (distances, indices)"""
    # 如果是一维 (768,) 需要 reshape
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1).astype('float32')
    else:
        query_vector = query_vector.astype('float32')
    
    distances, indices = index.search(query_vector, top_k)
    return distances[0], indices[0]

def main():
    # 1. 加载已经构建并保存好的索引 - 使用新的ViT-L/14索引
    index_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/index/faiss_index_ivf_vit_l14.index"
    index = load_faiss_index(index_path)
    print("索引已加载, 共包含向量数量:", index.ntotal)

    # 2. 加载元数据
    metadata_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/merged_metadata.csv"
    metadata_df = load_metadata(metadata_path)
    print("元数据加载完毕, 行数:", len(metadata_df))

    # 3. 准备一个查询向量 (这里随便拿第一张图片的向量做测试)
    vectors_path = "/Users/zhu_yangyang/Desktop/UkiyoeSearch_Project/ukiyoe_dataset/features/0000000000.npy"
    all_vectors = np.load(vectors_path).astype('float32')
    query_vector = all_vectors[0]

    # 4. 在索引中检索
    distances, indices = search_vectors(query_vector, index, top_k=5)
    print("检索到的Indices:", indices)
    print("对应的Distances:", distances)

    # 5. 映射回元数据
    #    比如 metadata_df.iloc[idx] 就是返回的第idx行，对应这张图片的元信息
    for rank, idx in enumerate(indices):
        row = metadata_df.iloc[idx]
        print(f"Rank {rank+1}, Index {idx}, Distance {distances[rank]}")
        print("元数据:", row.to_dict())   # 或者只打印你需要的列

if __name__ == "__main__":
    main()