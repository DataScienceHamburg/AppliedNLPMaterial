#%% packages
import re
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

# %%
lorem_ipsum = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Augue mauris augue neque gravida in fermentum et. Felis bibendum ut tristique et egestas quis ipsum suspendisse ultrices. Duis tristique sollicitudin nibh sit amet commodo nulla facilisi nullam. Pretium aenean pharetra magna ac placerat. Quis risus sed vulputate odio ut. Consectetur adipiscing elit duis tristique sollicitudin nibh. Nec nam aliquam sem et. Sed blandit libero volutpat sed cras. Faucibus pulvinar elementum integer enim neque volutpat ac. Mi in nulla posuere sollicitudin aliquam ultrices sagittis. Eget egestas purus viverra accumsan. Diam vel quam elementum pulvinar etiam non quam. Arcu cursus euismod quis viverra nibh cras. A scelerisque purus semper eget duis at. Lectus vestibulum mattis ullamcorper velit sed ullamcorper. Eget felis eget nunc lobortis mattis aliquam faucibus purus in. Elit scelerisque mauris pellentesque pulvinar pellentesque habitant. Ornare suspendisse sed nisi lacus sed. Interdum velit laoreet id donec ultrices. Ipsum a arcu cursus vitae congue mauris rhoncus aenean vel. Faucibus nisl tincidunt eget nullam non nisi. Urna condimentum mattis pellentesque id nibh. Tellus in hac habitasse platea dictumst vestibulum. Eget est lorem ipsum dolor. Enim eu turpis egestas pretium aenean pharetra magna ac placerat. Ac turpis egestas integer eget aliquet nibh. Vivamus arcu felis bibendum ut tristique et egestas. Nisi lacus sed viverra tellus in hac habitasse platea dictumst. Odio ut enim blandit volutpat maecenas volutpat. Turpis egestas sed tempus urna et pharetra pharetra massa. Dui nunc mattis enim ut tellus elementum sagittis vitae et. Nunc sed velit dignissim sodales ut eu. Aliquam ut porttitor leo a diam sollicitudin tempor id. At quis risus sed vulputate odio ut enim blandit volutpat. Gravida quis blandit turpis cursus in hac habitasse platea dictumst. Sit amet nulla facilisi morbi tempus iaculis urna. Diam maecenas sed enim ut sem viverra aliquet eget. Turpis egestas pretium aenean pharetra. At varius vel pharetra vel turpis nunc eget lorem. Integer quis auctor elit sed. Eget nunc lobortis mattis aliquam. Et magnis dis parturient montes nascetur ridiculus mus mauris vitae. Sollicitudin nibh sit amet commodo. Integer quis auctor elit sed vulputate mi sit amet mauris. Est placerat in egestas erat imperdiet. Ornare quam viverra orci sagittis eu volutpat odio facilisis mauris. Semper quis lectus nulla at volutpat diam. Amet volutpat consequat mauris nunc congue nisi. Ipsum nunc aliquet bibendum enim facilisis gravida neque convallis a. Et pharetra pharetra massa massa ultricies. Nunc eget lorem dolor sed viverra ipsum nunc aliquet bibendum."



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

# %%
lorem_ipsum_tokens = token_splitter.split_text(lorem_ipsum)
# %% max token length
max_token_length(lorem_ipsum_tokens)

# %% Real Implemetation
text_path = "../data/bible.txt"
with open(text_path, "r", encoding="utf-8") as f:
    text_raw = f.read()
# %%
text_raw[:1000]
# %% Character splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=['\n    \n', '\n\n', '\n', '. '],
    chunk_size=1000,
    chunk_overlap=0,
)
text_splitted = character_splitter.split_text(text_raw)
print(f"Total number of splitted chunks: {len(text_splitted)}")

max_token_length(text_splitted)


text_tokens = []
for text in text_splitted:
    text_tokens.extend(token_splitter.split_text(text))
print(f"Total number of tokens: {len(text_tokens)}")


# %% Check the token length
# reference: model card "By default, input text longer than 256 word pieces is truncated."
max_token_length(text_tokens)

# %%
embedding_fn = SentenceTransformerEmbeddingFunction()
# %% Size of embedding vector
len(embedding_fn([lorem_ipsum_tokens])[0])

# %%
chroma_db = chromadb.Client()
chroma_collection = chroma_db.create_collection("bible", embedding_function=embedding_fn)

# %% add all tokens to collection
ids = [str(uuid4()) for _ in range(len(text_tokens))]
chroma_collection.add(documents=text_tokens, ids=ids)

# %% Save the chroma collection
# %% Run a Query
res = chroma_collection.query(query_texts=["what did noah do?"], n_results=10)

res['documents']
# %%
