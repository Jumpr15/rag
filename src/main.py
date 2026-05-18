import pipeline.anime_dataset_2023
from pipeline.anime_dataset_2023 import map_ds

from qdrant_client import QdrantClient, models
import tqdm

client = QdrantClient(url="http://localhost:6333")

collection_name = "vector-store"
dense_vector_name = "dense"
sparse_vector_name = "sparse"
num_parallel_workers = 2

dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
sparse_model_name = "prithivida/Splade_PP_en_v1"


if not client.collection_exists(collection_name):
     client.create_collection(
          collection_name=collection_name,
          vectors_config={
               dense_vector_name: models.VectorParams(
                    size=client.get_embedding_size("sentence-transformers/all-MiniLM-L6-v2"),
                    distance=models.Distance.COSINE
               )
          },
          sparse_vectors_config={
               sparse_vector_name: models.SparseVectorParams()
          }
     )
     
documents = []
metadata = []

for row in map_ds:
     text = row['Synopsis']
     dense_embeds = models.Document(text=text, model=dense_model_name)
     sparse_embeds = models.Document(text=text, model=sparse_model_name)
     
     documents.append({
          dense_vector_name: dense_embeds,
          sparse_vector_name: sparse_embeds
     })
     metadata.append(row)
     
client.upload_collection(
     collection_name=collection_name,
     vectors=documents,
     payload=metadata,
     ids=tqdm.tqdm(range(len(documents)))
)

