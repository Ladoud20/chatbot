from src.embedding.sentenceTrans_embedding import SentenceTransEmbedding
from tqdm import tqdm
import numpy as np



class BatchEmbedding:
    def __init__(self, batchs, model_name_or_path, device='cpu'):
        self.batchs = batchs
        self.device = device
        self.embedding_model = SentenceTransEmbedding(model_name_or_path, self.device)


    def get_embeddings(self):
        batch_embeddings = []
        for _, batch in tqdm(enumerate(self.batchs), total=len(self.batchs), desc="Embedding chunks"):
            batch_embedding = self.embedding_model.encode(batch)
            batch_embeddings.append(batch_embedding)

        chunk_embeddings_np = np.vstack(batch_embeddings)
        return chunk_embeddings_np