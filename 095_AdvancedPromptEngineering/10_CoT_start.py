#%% packages
import openai
from openai import OpenAI
import pandas as pd
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
from pprint import pprint
# %%
openai_client = OpenAI()

# %% Function Definition
model_name = "gpt-3.5-turbo"

def chain_of_thought(user_query:str):
    pass

#%% Test
user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8."
res = chain_of_thought(user_query=user_prompt)
pprint(res)
# %%
