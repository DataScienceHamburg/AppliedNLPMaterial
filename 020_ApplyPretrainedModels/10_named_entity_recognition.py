#%% packages
from transformers import pipeline

# %% Named Entity Recognition
pipe = pipeline(task="ner")
pipe("Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in Cupertino, California.")


# %%
