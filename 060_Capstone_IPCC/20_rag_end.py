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

# %%
ipcc_report_file = "data/IPCC_AR6_WGII_TechnicalSummary.pdf"
reader = PdfReader(ipcc_report_file)
ipcc_texts = [page.extract_text().strip() for page in reader.pages]
# %%
pprint(ipcc_texts[5])
#%% filter out beginning and end of document
ipcc_texts_filt = ipcc_texts[5: -5]
print(f"Number of pages: {len(ipcc_texts_filt)}")
ipcc_texts_filt
# %% remove all header / footer texts
ipcc_wo_header_footer = [re.sub(r'\d+\nTechnical Summary', '', s) for s in ipcc_texts_filt]
# remove \nTS
ipcc_wo_header_footer = [re.sub(r'\nTS', '', s) for s in ipcc_wo_header_footer]

# remove TS\n
ipcc_wo_header_footer = [re.sub(r'TS\n', '', s) for s in ipcc_wo_header_footer]
ipcc_wo_header_footer


# %%
char_splitter = RecursiveCharacterTextSplitter(
    separators= ["\n\n", "\n", ". ", " ", ""],
    chunk_size=1000,
    chunk_overlap=0.2
    )

texts_char_splitted = char_splitter.split_text('\n\n'.join(ipcc_wo_header_footer))
print(f"Number of chunks: {len(texts_char_splitted)}")
# %%
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0.2,
    tokens_per_chunk=256
    )

texts_token_splitted = []
for text in texts_char_splitted:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except:
        print(f"Error in text: {text}")
        continue
    
print(f"Number of chunks: {len(texts_token_splitted)}")

# %%
pprint(texts_token_splitted[0])
# %% Vector Database
chroma_client = chromadb.PersistentClient(path="db")
chroma_collection = chroma_client.get_or_create_collection("ipcc")
# %% Add Documents
ids = [str(i) for i in range(len(texts_token_splitted))]
chroma_collection.add(
    ids=ids,
    documents=texts_token_splitted
    )
# %% query DB
query = "What is the impact of climate change on the ocean?"
res = chroma_collection.query(query_texts=[query], n_results=5)

# %%
def rag(query, n_results=5):
    res = chroma_collection.query(query_texts=[query], n_results=n_results)
    docs = res["documents"][0]
    joined_information = ';'.join([f'{doc}' for doc in docs])
    messages = [
        {
            "role": "system",
            "content": "You are a helpful expert on climate change. Your users are asking questions about information contained in attached information."
            "You will be shown the user's question, and the relevant information. Answer the user's question using only this information."
        },
        {"role": "user", "content": f"Question: {query}. \n Information: {joined_information}"}
    ]
    openai_client = OpenAI()
    model = "gpt-3.5-turbo"
    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content
# %%
query = "What is the impact of climate change on the ocean?"

rag(query=query, n_results=5)
# %%
