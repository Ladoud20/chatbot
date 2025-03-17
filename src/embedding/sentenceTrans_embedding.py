from sentence_transformers import SentenceTransformer
import torch



class SentenceTransEmbedding:
    def __init__(self, model_name_or_path, device = 'cpu'):
        self.model = SentenceTransformer(model_name_or_path, local_files_only=True)
        self.device = device
        self.model = self.model.to(self.device)

    def normalize_embeddings(self, embeddings):
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        normalized_embeddings = embeddings/norms
        return normalized_embeddings

    def encode(self,texts):
        embeddings = self.model.encode(texts, convert_to_tensor=True)
        if embeddings.dim() == 1:
            #single embedding, convert to 2D by adding an extra dimension (for consistency)
            embeddings = embeddings.unsqueeze(0)

        return self.normalize_embeddings(embeddings).cpu().numpy()