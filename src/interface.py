import gradio as gr

collection_name = "vector-store"

from search.dense_search import DenseSearcher
from search.sparse_search import SparseSearcher

def dense_search(text_query):
     searcher = DenseSearcher(collection_name)
     metadata = searcher.search(text_query)
     return metadata

def sparse_search(text_query):
     searcher = SparseSearcher(collection_name)
     metadata = searcher.search(text_query)
     return metadata

def vector_search(text_query):
     return "text"

demo = gr.Interface(
     fn=sparse_search,
     inputs=["text"],
     outputs=["text"],
)

demo.launch()