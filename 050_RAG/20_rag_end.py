#%% packages
import os
from pprint import pprint
import chromadb
import openai
from openai import OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
# chroma_db = chromadb.Client()  # on the fly
chroma_db = chromadb.PersistentClient(path="db")

#%%
chroma_db.list_collections()
#%% Get / Create Collection
chroma_collection = chroma_db.get_or_create_collection("movies")

res = chroma_collection.query(query_texts=["a monster in closet"], n_results=5)
#%% count of documents in collection
len(chroma_collection.get()['ids'])

# %% Run a Query
def get_query_results(query_text:str, n_results:int=5):
    res = chroma_collection.query(query_texts=[query_text], n_results=3)
    docs = res["documents"][0]
    titles = [item['title'] for item in res["metadatas"][0]]
    res_string = ';'.join([f'{title}: {description}' for title, description in zip(titles, docs)])
    return res_string

query_text = "a monster in the closet"
retrieved_results = get_query_results(query_text)

# %% RAG
system_role_definition = "You are a an expert in movies. Users will ask you questions about movies. You will get a user question, and relevant information. Relevant information is structured like movie title:movie plot; ... Please answer the question only using the information provided."
user_query = "What are the names of the movies and their plot where {user_query}?"
messages = [
    {"role": "system",
     "content": system_role_definition
     },
    {"role": "user",
     "content": f"{user_query}; \n Information: {retrieved_results}"
     }
]


# %%
openai_client = OpenAI()
model="gpt-3.5-turbo"
response = openai_client.chat.completions.create(
    model=model,
    messages=messages
)
    
# %%
content = response.choices[0].message.content
# %%
content
# %% bundle everything in a function
def rag(user_query:str):
    retrieved_results = get_query_results(user_query)
    system_role_definition = "You are a an expert in movies. Users will ask you questions about movies. You will get a user question, and relevant information. Relevant information is structured like movie title:movie plot; ... Please answer the question only using the information provided."
    user_query_complete = f"What are the names of the movies and their plot where {user_query}?"
    messages = [
        {"role": "system",
        "content": system_role_definition
        },
        {"role": "user",
        "content": f"{user_query_complete}; \n Information: {retrieved_results}"
        }
    ]
    openai_client = OpenAI()
    model="gpt-3.5-turbo"
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages
    )
    content = response.choices[0].message.content
    return content
    
# %% Response from Vector DB
print("Response from Vector DB")
print("-------------------------------------------------")
query = "a cop is chasing a criminal"
pprint(get_query_results(query))
#%%
print("Response from RAG")
print("-------------------------------------------------")
pprint(rag(query))

# %%
