import faiss
from src.indexing.faiss.base_faiss_index import BaseFaissIndex


class HNSWFaissIndex(BaseFaissIndex):

    def __init__(self, index_name, dimension, use_gpu=False, M=32, index_output_path='./indices/'):
        super().__init__(index_name, dimension, use_gpu, index_output_path)
        self.M = M

    def create_index(self,embeddings):
        print('Creating HNSW index...')
        self.index = faiss.IndexHNSWFlat(self.dimension, self.M)
        #self.move_to_gpu()
        self.index.add(embeddings)
        print('Index created!')
        #self.move_to_cpu()
        self.save_index()