#%% packages
from transformers import pipeline

# %% only provide task
pipe = pipeline(task="text-classification")
# %% run pipe
pipe("I like it very much.")
# %% provide model
pipe = pipeline(task="text-classification", 
                model="nlptown/bert-base-multilingual-uncased-sentiment")
# %% run pipe
# consume just a string
pipe("I like it very much.")

# %% consume a list
pipe(["I like it very much.", 
      "I hate it."])


# %%
