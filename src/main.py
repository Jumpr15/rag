import pipeline.anime_dataset_2023
from pipeline.anime_dataset_2023 import map_ds

from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://localhost:6333")

if not client.collection_exists("anime"):
     