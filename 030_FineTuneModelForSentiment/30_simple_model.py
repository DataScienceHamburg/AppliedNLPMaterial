#%% packages
from datasets import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModel, DistilBertTokenizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
# %% YELP Dataset
# source: https://huggingface.co/datasets/yelp_review_full/viewer/yelp_review_full/train?f%5blabel%5d%5bvalue%5d=0
yelp = load_dataset('yelp_review_full')
# %%
yelp

#%% create dataset
train_ds = yelp['train'].select(range(1000))


#%% Model and Tokenizer
model_name = 'distilbert-base-uncased'
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
model = AutoModel.from_pretrained(model_name).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# %% Tokenizer
text = 'Hello, this is a sample sentence!'

encoded_text = tokenizer(text, return_tensors='pt')
encoded_text
# %% Tokens
tokens = tokenizer.convert_ids_to_tokens(encoded_text['input_ids'][0])
# %% 
tokenizer.convert_tokens_to_string(tokens)

# %% how large is the vocabulary?
tokenizer.vocab_size

# %% Max context length
max_context_length = tokenizer.model_max_length
max_context_length
# %% Function for tokenization
def tokenize_text(batch):
    # padding...texts are filled with zeros based to longest example
    # truncation...texts are cut off after max_context_length
    return tokenizer(batch['text'], return_tensors='pt', padding='max_length', truncation=True)
# %%
yelp_encodings = train_ds.map(tokenize_text, batched=True, batch_size=128)
# %%
yelp_encodings.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])  # encodings need to be converted to torch tensors
def get_last_hidden_state(batch):
    inputs = {k: v for k, v in batch.items() if k in tokenizer.model_input_names}
    with torch.no_grad():
        last_hidden_state = model(**inputs).last_hidden_state
        # [:, 0] refers to CLS token for complete sentence representation
    return {'hidden_state': last_hidden_state[:, 0]}
# %%
yelp_hidden_states = yelp_encodings.map(get_last_hidden_state, batched=True, batch_size=128)  # will have additional column 'hidden_state'

# %%
yelp_hidden_states

#%%
import joblib
joblib.dump(yelp_hidden_states, 'model/yelp_hidden_states.joblib')


#%%
cutoff = 800
X_train = np.array(yelp_hidden_states['hidden_state'][:cutoff])
y_train = np.array(yelp_hidden_states['label'][:cutoff])
X_test = np.array(yelp_hidden_states['hidden_state'][cutoff: ])
y_test = np.array(yelp_hidden_states['label'][cutoff: ])
print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

#%% Dummy model
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
dummy_model.score(X_test, y_test)

# %% SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)
svm_model.score(X_test, y_test)

# %% Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test)

# %%

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_model.score(X_test, y_test)
# %%
