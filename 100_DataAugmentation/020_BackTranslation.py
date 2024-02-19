#%% packages
# !pip install sentencepiece 
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#%% Model and Tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
# %% from original to target language
original_en = 'The pen is mightier than the sword.'
original_encoded = tokenizer(original_en, return_tensors="pt")
generated_tokens = model.generate(**original_encoded, forced_bos_token_id=tokenizer.get_lang_id("de"))
target_de = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
target_de
# %% from target back to original language
target_encoding = tokenizer(target_de, return_tensors="pt", padding=True)
generated_tokens = model.generate(**target_encoding, forced_bos_token_id=tokenizer.get_lang_id("en"))

tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# %%
