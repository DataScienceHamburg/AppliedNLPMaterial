#%% packages
import os
import numpy as np
from matplotlib import pyplot as plt

import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import open_clip

#%% load model
# source: https://pypi.org/project/open-clip-torch-any-py3/
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32-quickgelu', pretrained='laion400m_e32')

#%% prepare vector db
chroma_db = chromadb.Client()

img_loader = ImageLoader()
multimodal_embedding_fn = OpenCLIPEmbeddingFunction()
chroma_collection = chroma_db.get_or_create_collection("dogs", embedding_function=multimodal_embedding_fn, data_loader=img_loader)
# %% add images to DB
img_folder = "../data/dogs"
img_files = os.listdir(img_folder)
img_files = [f"{img_folder}/{img_file}" for img_file in img_files]
chroma_collection.add( ids=img_files, documents=img_files, uris=img_files)

# %% helper function to show query results
def show_query_results(query_list:list, query_result:dict):
    res_count = len(query_result['ids'][0])
    for i in range(len(query_list)):
        print(f"Query: {query_list[i]}")
        for j in range(res_count):
            id = query_result['ids'][i][j]
            distance = query_result['distances'][i][j]
            data = query_result['data'][i][j]
            uri = query_result['uris'][i][j]
            print(f"Result {j}: {uri} with distance: {np.round(distance, 2)}")
            plt.imshow(data)
            plt.axis("off")
            plt.show()
# %% Query Text
query_list = ["dog in grassland"]
# query_list = ["dog in black fur"]
# query_list = ["dog, water"]
# query_list = ["akita"]
query_result = chroma_collection.query(query_texts = query_list, 
                                       n_results=3, 
                                       include=['documents', 'distances', 'metadatas', 'data', 'uris'],)
# %% Query Image
query_list = ["../data/dogs/akita_1.jpg"]
query_result = chroma_collection.query(query_images=query_list, n_results=2, include=['documents', 'distances', 'metadatas', 'data', 'uris'],)

# %%
show_query_results(query_list, query_result)