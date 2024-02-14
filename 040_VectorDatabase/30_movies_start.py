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

#%% filter movies for missing title or overwiew

# filter for unique ids


#%%


# %%

#%% Word Distribution


#%% visualize token distribution

# %%

# %%
# chroma_db = chromadb.Client()  # on the fly
# persistent

#%% List Collections

#%% Get / Create Collection

# %% add all tokens to collection

#%% Add movies to vector DB


#%% count of documents in collection


# %% Function to get title


#%% Test the function


# %%
