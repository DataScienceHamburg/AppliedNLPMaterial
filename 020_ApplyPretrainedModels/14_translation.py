#%% packages
from transformers import pipeline



#%% translation
pipe = pipeline("translation_en_to_de")
pipe("The capital of France is Paris.")
# %%
