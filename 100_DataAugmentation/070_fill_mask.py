#%%
from transformers import pipeline

#%%
pipe = pipeline('fill-mask')
pipe('This coffee is very <mask>.')
# %%