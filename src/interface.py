import gradio as gr
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")

collection_name = "vector-store"

from search.dense_search import DenseSearcher
from search.sparse_search import SparseSearcher
from search.hybrid_search import HybridRRFSearcher

def dense_search(text_query):
     searcher = DenseSearcher(collection_name)
     metadata = searcher.search(text_query)
     return metadata

def sparse_search(text_query):
     searcher = SparseSearcher(collection_name)
     metadata = searcher.search(text_query)
     return metadata

def hybrid_rrf_search(text_query):
     searcher = HybridRRFSearcher(collection_name)
     metadata = searcher.search(text_query)
     return metadata

def vector_search(text_query):
     return "text"

demo = gr.Interface(
     fn=hybrid_rrf_search,
     inputs=["text"],
     outputs=["text"],
)

demo.launch()