#%% packages
import openai
from openai import OpenAI
import pandas as pd
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
from pprint import pprint

openai_client = OpenAI()

# %% Function Definition CoT
model_name = "gpt-3.5-turbo"

def chain_of_thought(user_query:str):
    system_role_definition = "You are a helpful assistent and answer precise and concise."
    user_query_complete = f"{user_query}, think step-by-step."
    messages = [
        {"role": "system",
         "content": system_role_definition
         },
        {"role": "user",
         "content": f"{user_query_complete}"
         }
    ]
    response = openai_client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    content = response.choices[0].message.content
    return content

#%% Function Definition SelfConsistency CoT
def self_consistency_chain_of_thought(user_query:str, number_of_chains:int=3):
    # run CoT multiple times
    pass

#%% Test
user_prompt = "The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. The numbers are 3, 4, 6, and 8."
res = self_consistency_chain_of_thought(user_query=user_prompt)
pprint(res)
# %%