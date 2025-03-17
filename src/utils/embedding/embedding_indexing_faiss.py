from src.preprocessing.data_batching import DataBatching
from src.embedding.batch_embedding import BatchEmbedding
from src.indexing.faiss.HNSW_faiss_index import HNSWFaissIndex
from sentence_transformers import SentenceTransformer

class EmbeddingIndexingFAISS:
    def __init__(self, df, text_column, model_name_or_path, num_batchs, index_name, index_type, index_output_path, device="cpu"):
        self.df = df
        self.texts = self.df[text_column].tolist()
        self.model_name_or_path = model_name_or_path
        self.num_batchs = num_batchs
        self.index_name = index_name
        self.index_type = index_type
        self.index_output_path = index_output_path
        self.device = device

    def create_index(self):
        dataBatching = DataBatching(self.texts, num_batchs=self.num_batchs)
        chunks = dataBatching.batch()
        # Get the embedding with CUDA device
        batchEmbedding = BatchEmbedding(chunks,self.model_name_or_path,device=self.device)
        embeddings = batchEmbedding.get_embeddings()
        # Add the vectors to the column text_vector
        self.df['text_vector'] = list(embeddings)
        # Create FAISS index
        self.dimension = embeddings.shape[1]
        self.use_gpu = True if self.device=="cpu" else False
        HNSW_index = HNSWFaissIndex(index_name=self.index_name,
                                        dimension=self.dimension,
                                        use_gpu=self.use_gpu,
                                        index_output_path=self.index_output_path)
        HNSW_index.create_index(embeddings)




