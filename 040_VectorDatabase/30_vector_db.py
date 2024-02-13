#%% packages
import re
import pandas as pd
import numpy as np
import seaborn as sns
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from uuid import uuid4

# %% max_length
def max_token_length(txt_list:list):
    max_length = 0
    for txt in txt_list:
        token_count = len(re.findall(r'\w+', txt))
        if token_count > max_length:
            max_length = token_count
    return f"Max Token Length: {max_length} tokens"

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
text_path = "data/movies.csv"
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
text_tokens = []
for text in df_movies_filt.loc[:, "overview"]:
    text_tokens.extend(token_splitter.split_text(text))
print(f"Total number of chunks: {len(text_tokens)}")

# %%
max_token_length(text_tokens)

#%% Token Distribution
token_count = []
for txt in df_movies_filt.loc[:, "overview"]:
    token_count.append(len(re.findall(r'\w+', txt)))

#%% visualize token distribution
sns.histplot(token_count, bins=100)

# %%
embedding_fn = SentenceTransformerEmbeddingFunction()

# %%
# chroma_db = chromadb.Client()  # on the fly
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
def get_query_results(query_text:str):
    res = chroma_collection.query(query_texts=[query_text], n_results=1)
    print(f"Title: {res['metadatas'][0][0]['source']} \n")
    print(f"Description: {res['documents'][0][0]} \n")
    
    
    
get_query_results("a monster in the closet")

# %%
res['metadatas'][0][0]['source']
# %%
