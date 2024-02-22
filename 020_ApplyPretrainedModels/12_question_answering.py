#%% packages
from transformers import pipeline



# %% Question Answering
pipe = pipeline("question-answering")
pipe(context="The Big Apple is a nickname for New York City.", question="What is the Big Apple?")

# %%
