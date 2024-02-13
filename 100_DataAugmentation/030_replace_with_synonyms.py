#%% packages
import nlpaug.augmenter.word as naw

# %%
text = "The quick brown fox jumps over the lazy dog."
aug = naw.SynonymAug(aug_src='wordnet')
augmented_text = aug.augment(text)
print("Original:")
print(text)
print("Augmented Text:")
print(augmented_text)
# %%
