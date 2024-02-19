#%%
from transformers import pipeline

#%%
pipe = pipeline('fill-mask')
pipe('The pen is <mask> than the sword.')
# %%