import re
from rank_bm25 import BM25Okapi
from stop_words import get_stop_words

class BM25Corpus:
    def __init__(self, texts, language):
        self.texts = texts
        self.language = language
        if self.language == 'de':
            self.stopwords = get_stop_words('de')
        elif self.language == 'en':
            self.stopwords = get_stop_words('en')
        else:
            raise ValueError("Unsupported language: choose 'de' for German or 'en' for English")

    def clean_token(self):
        cleaned_texts = []
        for text in self.texts:
            if not isinstance(text, str):  # Ensure text is a string
                text = str(text)

            cleaned_tokens = []
            for token in text.lower().split():
                token = re.sub(r'[^\w\s]', '', token)  # Remove punctuation
                if len(token) > 0 and token not in self.stopwords:
                    cleaned_tokens.append(token)
            cleaned_texts.append(cleaned_tokens)

        self.cleaned_texts = cleaned_texts

    def create_corpus(self):
        if not hasattr(self, 'cleaned_texts'):
            raise ValueError("Call clean_token() before creating the corpus.")
        return BM25Okapi(self.cleaned_texts)
