from rank_bm25 import BM25Okapi
from stop_words import get_stop_words
import re
import numpy as np

class BM25_search:
    def __init__(self, bm25, language):
        self.bm25 = bm25
        self.language = language
        if self.language == 'de':
            self.stopwords = get_stop_words('de')
        elif self.language == 'en':
            self.stopwords = get_stop_words('en')
        else:
            raise ValueError("Unsupported language: choose 'de' for German or 'en' for English")

    def clean_query(self, query):
        cleaned_tokens = []
        # Split sentence into words
        for token in query.lower().split():
            # Remove all punctuations (both leading, trailing, and inside words)
            token = re.sub(r'[^\w\s]', '', token)  # Remove punctuation
            # Filter out stop words and empty tokens
            if len(token) > 0 and token not in self.stopwords:
                cleaned_tokens.append(token)
        self.cleaned_query = cleaned_tokens
        return cleaned_tokens

    def search(self, query, top_k=5):
        cleaned_query = self.clean_query(query)
        scores = self.bm25.get_scores(cleaned_query)
        # Get the indices of the top-k documents
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_scores = scores[top_indices]
        return top_scores, top_indices