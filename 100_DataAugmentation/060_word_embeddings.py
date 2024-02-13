#%% packages
import nlpaug.augmenter.word as naw

# !pip install gensim
# %%
augmentation = naw.WordEmbsAug(model_type='glove', model_path='glove.6B.100d.txt', action="substitute")
# %%
text_original = 'The pen is mightier than the sword.'
augmentation.augment(text_original)
# %%
