#%% packages
import os
import re
from pprint import pprint
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
import chromadb
import openai
from openai import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]

# %% Text Extraction



# %% Show first page
#%% filter out beginning and end of document

# %% remove all header / footer texts
# remove "Technical Summary" and the number before it

# remove \nTS

# remove TS\n

# %% Split Text

# %% Token Split


# %% Show first chunk

# %% Vector Database



# %% Add Documents


# %% query DB


# %% RAG


# %% Test RAG

# %%
