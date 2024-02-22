#%% packages
from transformers import pipeline


#%% fill-mask
pipe = pipeline("fill-mask")
pipe("The capital of France is <mask>.")
# %%
