from sentence_transformers import CrossEncoder

class CEReranker:
    def __init__(self, model_name_or_path, device='cpu'):
        self.model = CrossEncoder(model_name_or_path)
        self.device = device

    def reranker(self, query, doc_candidates, candidates_idx, top_n=5):
        pairs = []
        for doc in doc_candidates:
            pair = [query, doc]
            pairs.append(pair)
        scores = self.model.predict(pairs)
        score_idx_pairs = list(zip(scores, candidates_idx))
        sorted_scores_with_idx = sorted(score_idx_pairs, key=lambda x: x[0], reverse=True)

        sorted_scores = [score for score, _ in sorted_scores_with_idx[:top_n]]
        sorted_indices = [idx for _, idx in sorted_scores_with_idx[:top_n]]
        return sorted_scores, sorted_indices
