from qdrant_client import QdrantClient, models

class SparseSearcher:
     def __init__(self, collection_name):
          self.sparse_model_name = "prithivida/Splade_PP_en_v1"
          self.sparse_vector_name = "sparse"
          self.collection_name = collection_name
          self.qdrant_client = QdrantClient(url="http://localhost:6333")
          
     def search(self, text: str):
          res = self.qdrant_client.query_points(
               collection_name=self.collection_name,
               query=models.Document(text=text, model=self.sparse_model_name),
               using=self.sparse_vector_name,
               query_filter=None,
               limit=1
          ).points
          
          return [point.payload for point in res]
     
if __name__ == "__main__":
     collection_name = "vector-store"
     query = "skibidi toilet"
     
     searcher = SparseSearcher(collection_name)
     metadata = searcher.search(query)
     print(metadata)