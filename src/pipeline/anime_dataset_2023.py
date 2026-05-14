import kagglehub
from kagglehub import KaggleDatasetAdapter

from sentence_transformers import SentenceTransformer

dataset = "dbdmobile/myanimelist-dataset"
file_path = "anime-dataset-2023.csv"

hf_dataset = kagglehub.dataset_load(
     KaggleDatasetAdapter.HUGGING_FACE,
     dataset,
     file_path
)

# splits comma seperated genres into array
def split_genres_by_commas(row):
     genres = row['Genres']
     genre_list = genres.split(", ")
     row['Genres'] = genre_list
     return row

dense_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# encodes synopsis as dense embeddings 
def encode_to_dense_embeddings(row):
     synopsis = row['Synopsis']
     dense_embedding = model.encode(synopsis) # embeddings of (384,)
     row['Synopsis'] = dense_embedding
     return row

# drops all columns except id, english name, score, genre, synopsis
ds = hf_dataset.select_columns(['anime_id', 'English name', 'Score', 'Genres', 'Synopsis'])

# slice so my cpu doesnt explode
ds = ds.select(range(3))

map_ds = ds.map(split_genres_by_commas)

map_ds = map_ds.map(encode_to_dense_embeddings)



print(map_ds)
print(map_ds[0])