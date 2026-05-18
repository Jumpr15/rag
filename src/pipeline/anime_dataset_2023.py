import kagglehub
from kagglehub import KaggleDatasetAdapter

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

# drops all columns except id, english name, score, genre, synopsis
ds = hf_dataset.select_columns(['anime_id', 'English name', 'Score', 'Genres', 'Synopsis'])

# slice so my cpu doesnt explode
ds = ds.select(range(3))

map_ds = ds.map(split_genres_by_commas)
