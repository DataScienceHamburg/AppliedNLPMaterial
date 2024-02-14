#%% packages
import re
import pandas as pd
import seaborn as sns
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pprint import pprint
# %% max_length
def max_word_count(txt_list:list):
    max_length = 0
    for txt in txt_list:
        word_count = len(re.findall(r'\w+', txt))
        if word_count > max_length:
            max_length = word_count
    return f"Max Word Count: {max_length} words"

# %% Sentence splitter
# chroma default sentence model "all-MiniLM-L6-v2"
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
# max input length: 256 characters
model_max_chunk_length = 256
token_splitter = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=model_max_chunk_length,
    model_name="all-MiniLM-L6-v2",
    chunk_overlap=0
)

# %% Data Import
text_path = "../data/movies.csv"
df_movies_raw = pd.read_csv(text_path, parse_dates=['release_date'])

df_movies_raw.head(2)

#%% filter movies for missing title or overwiew
selected_cols = ['id', 'title', 'overview', 'vote_average', 'release_date']
df_movies_filt = df_movies_raw[selected_cols].dropna()
# filter for unique ids
df_movies_filt = df_movies_filt.drop_duplicates(subset=['id'])

# filter for movies after 01.01.2023
df_movies_filt = df_movies_filt[df_movies_filt['release_date'] > '2023-01-01']
df_movies_filt.shape
#%%


# %%
max_word_count(df_movies_filt['overview'])

#%% Word Distribution
descriptions_len = []
for txt in df_movies_filt.loc[:, "overview"]:
    descriptions_len.append(len(re.findall(r'\w+', txt)))

#%% visualize token distribution
sns.histplot(descriptions_len, bins=100)

# %%
embedding_fn = SentenceTransformerEmbeddingFunction()

# %%
# chroma_db = chromadb.Client()  # on the fly
# persistent
chroma_db = chromadb.PersistentClient(path="db")

#%%
chroma_db.list_collections()
#%% Get / Create Collection
# chroma_collection = chroma_db.create_collection("movies", embedding_function=embedding_fn)
# chroma_collection = chroma_db.get_collection("movies")
chroma_collection = chroma_db.get_or_create_collection("movies")
# %% add all tokens to collection
ids = [str(i) for i in df_movies_filt['id'].tolist()]
documents = df_movies_filt['overview'].tolist()
titles = df_movies_filt['title'].tolist()
metadatas = [{'source': title} for title in titles]
#%%
chroma_collection.add(documents=documents, ids=ids, metadatas=metadatas)

#%% count of documents in collection
len(chroma_collection.get()['ids'])

# %% Save the chroma collection
# %% Run a Query
def get_title_by_description(query_text:str):
    n_results = 3
    res = chroma_collection.query(query_texts=[query_text], n_results=n_results)
    for i in range(n_results):
        pprint(f"Title: {res['metadatas'][0][i]['source']} \n")
        pprint(f"Description: {res['documents'][0][i]} \n")
        pprint("-------------------------------------------------")

#%% Run a Query
get_title_by_description(query_text="monster, underwater")

# %%
