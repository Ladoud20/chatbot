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
data_path = "../../data/insurance_claims.csv"
df = pd.read_csv(data_path)

# Create an overview column, that will be used for retrieval
df['overview'] = (
        'Policy Number: ' + df['policy_number'].astype(str) + '\n' +
        'Customer Age: ' + df['age'].astype(str) + '\n' +
        'Policy State: ' + df['policy_state'] + '\n' +
        'Annual Premium: ' + df['policy_annual_premium'].astype(str) + '\n' +
        'Vehicle: ' + df['auto_make'] + ' ' + df['auto_model'] + ' (' + df['auto_year'].astype(str) + ')\n' +
        'Incident Date: ' + df['incident_date'] + '\n' +
        'Incident Type: ' + df['incident_type'] + '\n' +
        'Severity: ' + df['incident_severity'] + '\n' +
        'Total Claim Amount: ' + df['total_claim_amount'].astype(str) + '\n' +
        'Fraud Reported: ' + df['fraud_reported']
)

df['index'] = df.index

df = df[['index', 'policy_number', 'age', 'policy_state', 'auto_make', 'auto_model', 'incident_date', 'incident_type', 'total_claim_amount', 'fraud_reported', 'overview']]

# Create a text list from overview column
# This will be used for retrieval
texts = df['overview'].tolist()

# Create a metadata dict about the information we want to return
# We will return the auto_model, incident_date and total_claim_amount
metadatas = ['auto_model', 'incident_date', 'total_claim_amount']



embedding_model_path = "../../models/sentence_transformer_en"
model = SentenceTransEmbedding(embedding_model_path, device='cpu')

index_path = "../../data/database/faiss"
index_name = "HNSW_car_index.index"
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
    return templates.TemplateResponse("index_car.html", {"request": request})

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
         top_n=2
     )
    results = []
    for i, result in enumerate(top_n_index):

        idx = int(top_n_index[i])



        metadata_info = {
            "auto_model": df.loc[idx, 'auto_model'],
            "incident_date": df.loc[idx, 'incident_date'],
            "total_claim_amount": int(df.loc[idx, 'total_claim_amount'])
        }
        print(metadata_info)
        results.append(metadata_info)

    print(top_n_index)
    print(results)
    #use prompt to get the final answer
    """
    prompt = here I have to write what I want to get by providing the query and the final results from reranker
    openai.api_key = api_key
    answer = get_gpt_response(prompt)
    """
    return templates.TemplateResponse("index_car.html", {"request": request, "query": query, "results": results})



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port= 8001)