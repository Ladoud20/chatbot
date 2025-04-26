import numpy as np
import pandas as pd
import time
import os
from src.utils.embedding.embedding_indexing_faiss import EmbeddingIndexingFAISS
from src.embedding.sentenceTrans_embedding import SentenceTransEmbedding
from src.retrieving.faiss_search import FaissSearch

# Load the data
data_path = '../data/insurance_claims.csv'
df = pd.read_csv(data_path)


df['index'] = df.index


# Combine relevant columns into a "description" for embedding
df['description'] = df.apply(lambda row: f"Incident: {row['incident_type']}, Collision: {row['collision_type']}, "
                                         f"Severity: {row['incident_severity']}, Damage: {row['property_damage']}, "
                                         f"Car: {row['auto_make']} {row['auto_model']} ({row['auto_year']})", axis=1)


#print("Sample description:", df['description'].iloc[0])


embedding_model_path = "../models/sentence_transformer_en"
index_output_path = '../indices'


EmbIndFAISS = EmbeddingIndexingFAISS(
    df=df,
    text_column="description",
    model_name_or_path=embedding_model_path,
    num_batchs=len(df)/500,
    index_name= "car_inssurance.index",
    index_type="HNSW",
    index_output_path=index_output_path,
    device="cpu"
)

EmbIndFAISS.create_index()