from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
from src.indexing.bm25.bm_25 import BM25Corpus
from src.retrieving.bm25_search import BM25_search
from src.reranking.ce_reranker import CEReranker
from src.embedding.sentenceTrans_embedding import SentenceTransEmbedding
from src.utils.embedding.embedding_indexing_faiss import EmbeddingIndexingFAISS
from src.retrieving.faiss_search import FaissSearch



# Load the csv file
data_path = "../../data/books.csv"
df = pd.read_csv(data_path)

# Create an overview column, that will be used for retrieval
df['overview'] = (
        'Title: ' + df['title'] + '\n' +
        'Authors: ' + df['authors'] + '\n' +
        'Categories: ' + df['categories'] + '\n' +
        'Description: ' + df['description']
)

df['index'] = df.index

df = df[['index', 'title', 'authors', 'categories', 'description', 'overview']]

# Create a text list from overview column
# This will be used for retrieval
texts = df['overview'].tolist()

# Create a metadata dict about the information we want to return
# We will return the title, author and description
metadatas = ['title', 'authors', 'description']



embedding_model_path = "../../models/sentence_transformer_en"
model = SentenceTransEmbedding(embedding_model_path, device='cpu')

index_path = "../../data/database/faiss"
index_name = "HNSW_book_index.index"
index_type = "HNSW"

HNSW_search = FaissSearch(index_name=index_name,
                          index_type=index_type,
                          use_gpu=False,
                          index_path=index_path)

HNSW_search.load_index()

bm25_corpus = BM25Corpus(texts, language='en')

#tokenizer
corpus = bm25_corpus.clean_token()

# create a corpus
bm25 = bm25_corpus.create_corpus()

reranker_name_or_path = "../../models/reranking_models/cross-encoder-en"
reranker = CEReranker(model_name_or_path=reranker_name_or_path)

app = FastAPI()
templates = Jinja2Templates(directory="../templates")

app.mount("/static", StaticFiles(directory="../static"), name="static")
api_key = "need to add it"

class QueryRequest(BaseModel):
    text: str

@app.get("/hello")
def read_root():
    return {"message" : "Hello Khalil!"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search(request: Request, query: str= Form(...)):
    query_embedding = model.encode(query)
    _, semantic_indices = HNSW_search.search(query_embedding, top_k=10)
    semantic_indices = semantic_indices.reshape(-1)
    BM25search = BM25_search(bm25=bm25, language='en')
    _, bm25_indices = BM25search.search(query,top_k=10)


    candidates_idx = set(list(semantic_indices) + list(bm25_indices))
    doc_candidates = []

    for _, idx in enumerate(candidates_idx):
        doc = df.loc[idx, 'overview']
        doc_candidates.append(doc)

    _, top_n_index = reranker.reranker(
        query=query,
         doc_candidates=doc_candidates,
         candidates_idx=candidates_idx,
         top_n=5
     )
    results = []
    for i, result in enumerate(top_n_index):
        idx = int(top_n_index[i])
        metadata_info = {
            "title": df.loc[idx, 'title'],
            "authors": df.loc[idx, 'authors'],
            "description": df.loc[idx, 'description']
        }
        results.append(metadata_info)

        #use prompt to get the final answer
        """
        prompt = here I have to write what I want to get by providing the query and the final results from reranker
        openai.api_key = api_key
        answer = get_gpt_response(prompt)
        """
        return templates.TemplateResponse("index.html", {"request": request, "query": query, "results": results})



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port= 8001)