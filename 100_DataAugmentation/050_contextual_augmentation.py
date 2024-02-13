#%% packages
import nlpaug.augmenter.word as naw

# %%
model_name = 'bert-base-uncased'
augmentation = naw.ContextualWordEmbsAug(model_path=model_name, action="substitute", device='cuda')
text_original = 'The pen is mightier than the sword.'
augmentation.augment(text_original)
# %%
