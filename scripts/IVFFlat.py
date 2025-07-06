import faiss
import numpy as np

# 1. Define or load your vectors:
# Option A: Load from a .npy file
# vectors = np.load('my_vectors.npy')   # shape must be (58000, 512), for example

# Option B: Generate random vectors for testing
vectors = np.random.random((58000, 512))

# 2. Convert to float32 if necessary
vectors_f32 = vectors.astype('float32')

d = vectors_f32.shape[1]  # 512
index = faiss.IndexFlatL2(d)  # A flat (L2) index

# 3. Build the index
index.add(vectors_f32)
print("向量数量:", index.ntotal)

# 4. Simple search example
query_vector = vectors_f32[0:1]  # take the first vector as the query
k = 5  # retrieve 5 nearest neighbors
D, I = index.search(query_vector, k)
print("索引号 (I):", I)
print("距离 (D):", D)

