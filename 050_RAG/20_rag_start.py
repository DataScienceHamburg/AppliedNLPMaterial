#%% packages
import os
from pprint import pprint
import chromadb
import openai
from openai import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
# chroma_db = chromadb.Client()  # on the fly
chroma_db = chromadb.PersistentClient(path="db")

#%% List Collections

#%% Get / Create Collection



#%% count of documents in collection


# %% Query Function




# %% RAG setup


# %% Pass Vector DB response to RAG

    
# %% Response from RAG

# %%

# %% bundle everything in a function

    
# %% Response from Vector DB
print("Response from Vector DB")
print("-------------------------------------------------")
query = "a cop is chasing a criminal"

#%%
print("Response from RAG")
print("-------------------------------------------------")


