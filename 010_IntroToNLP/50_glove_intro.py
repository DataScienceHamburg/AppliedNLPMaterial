#%%
import torch
import torchtext.vocab as vocab 
# %% Code Change because of GloVe download issue
# solution provided by Mitchell Miller
# glove = vocab.GloVe(name='6B', dim =100)
from torchtext.vocab import Vectors
 
class newGloVe(Vectors):
    url = {
        "42B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.42B.300d.zip", 
        "840B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip",
        "twitter.27B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.twitter.27B.zip",
        "6B": "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    }
 
    def __init__(self, name="840B", dim=300, **kwargs) -> None:
        url = self.url[name]
        print(f"Downloading from {url}")
        name = "glove.{}.{}d.txt".format(name, str(dim))
        super(newGloVe, self).__init__(name, url=url, **kwargs)
 
glove = newGloVe(name='6B', dim=100)

# %% number of words and embeddings
glove.vectors.shape

#%% get an embedding vector
def get_embedding_vector(word):
    word_index = glove.stoi[word]
    emb = glove.vectors[word_index]
    return emb

get_embedding_vector('chess').shape

#%% find closest words from input word
def get_closest_words_from_word(word, max_n=5):
    word_emb = get_embedding_vector(word)
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return dist_sort_filt

get_closest_words_from_word('chess')

#%% find closest words from embedding
def get_closest_words_from_embedding(word_emb, max_n=5):
    distances = [(w, torch.dist(word_emb, get_embedding_vector(w)).cpu().item()) for w in glove.itos]
    dist_sort_filt = sorted(distances, key=lambda x: x[1])[:max_n]
    return dist_sort_filt
# %% find word analogies
# e.g. King is to Queen like Man is to Woman
def get_word_analogy(word1, word2, word3, max_n=5):
    # logic w1= king, ...
    # w2 - w1 + w3 --> w4
    word1_emb = get_embedding_vector(word1)
    word2_emb = get_embedding_vector(word2)
    word3_emb = get_embedding_vector(word3)
    word4_emb = word2_emb - word1_emb + word3_emb
    analogy = get_closest_words_from_embedding(word4_emb)
    return analogy

get_word_analogy(word1='sister', word2='brother', word3='nephew')
    
    
