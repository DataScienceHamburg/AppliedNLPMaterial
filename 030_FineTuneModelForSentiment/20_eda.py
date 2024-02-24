#%% packages
from huggingface_hub import list_datasets
from datasets import load_dataset
import pandas as pd
import seaborn as sns

# %%
datasets = list_datasets()
for dataset in datasets:
    print(dataset)

# %% YELP Dataset
# source: https://huggingface.co/datasets/yelp_review_full/viewer/yelp_review_full/train?f%5blabel%5d%5bvalue%5d=0
yelp = load_dataset('yelp_review_full')
# %%
yelp

#%% create dataset
train_ds = yelp['train']
# %%
train_ds[0]
# %%
train_ds.features
# %% convert to dataframe (if necessary)
# train_ds.set_format('pandas')
# train_ds[:]
# %% imbalance of dataset
val_count = pd.DataFrame(train_ds['label']).value_counts()

sns.countplot(val_count.tolist())


# %% understanding the texts
# count of words per review / per class
df_review_len_label = pd.DataFrame({'review_length': [len(s.split()) for s in train_ds['text']], 'label': train_ds['label']})
sns.boxplot(x='label', y='review_length', data=df_review_len_label)


# %%
