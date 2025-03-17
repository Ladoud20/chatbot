import faiss
import numpy as np


class FaissSearch:
    def __init__(self, index_name, index_type="HNSW", use_gpu=False, index_path="./indices/"):
        self.index_name = index_name
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.index_path = index_path

    def load_index(self):
        index_file = self.index_path + self.index_name
        print(f"Loading index from {index_file}")

        self.index = faiss.read_index(index_file)

        # Ensure index is searched using CPU
        if self.use_gpu:
            print("Warning: GPU is not yet enabled. Running on CPU.")

        print("Index loaded successfully!")

    def search(self, query_embedding, top_k=5):
        if self.index is None:
            raise ValueError("Index not loaded. Please load the index first using load_index().")

        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding).astype("float32")

        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k)
        return distances, indices
