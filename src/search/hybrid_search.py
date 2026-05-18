from qdrant_client import QdrantClient, models

class HybridRRFSearcher:
     def __init__(self, collection_name):
          self.dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
          self.sparse_model_name = "prithivida/Splade_PP_en_v1"
          self.dense_vector_name = "dense"
          self.sparse_vector_name = "sparse"

          self.collection_name = collection_name
          self.qdrant_client = QdrantClient(url="http://localhost:6333")
          
     def search(self, text: str):
          res = self.qdrant_client.query_points(
               collection_name=self.collection_name,
               query=models.FusionQuery(
                    fusion=models.Fusion.RRF
               ),
               prefetch=[
                    models.Prefetch(
                         query=models.Document(text=text, model=self.dense_model_name),
                         using=self.dense_vector_name
                    ),
                    models.Prefetch(
                         query=models.Document(text=text, model=self.sparse_model_name),
                         using=self.sparse_vector_name
                    )
               ],
               query_filter=None,
               limit=1
          ).points
          
          return [point.payload for point in res]
     
if __name__ == "__main__":
     collection_name = "vector-store"
     query = "skibidi toilet"
     
     searcher = HybridRRFSearcher(collection_name)
     metadata = searcher.search(query)
     print(metadata)