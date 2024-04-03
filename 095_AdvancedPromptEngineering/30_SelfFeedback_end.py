#%% packages
import openai
from openai import OpenAI
import pandas as pd
import os
openai.api_key = os.environ["OPENAI_API_KEY"]
from pprint import pprint
import json
import re
# %%
openai_client = OpenAI()

# %%
def self_feedback(user_prompt:str, max_iterations:int=3, target_rating:int=90):
    model_name = "gpt-3.5-turbo"
    system_prompt = """An input is provided in <<>>. Evaluate the input in terms of how well it addresses the original task of explaining the key events and significance of the American Civil War. Consider factors such as: Breadth and depth of context provided; Coverage of major events; Analysis of short-term and long-term impacts/consequences; If you identify any gaps or areas that need further elaboration: Return output as json {"rating": "scoring in percentage", "feedback": "detailed feedback","revised_output": "return an improved output describing the key events and significance of the American Civil War. Avoid special characters like apostrophes (') and double quotes (\")"}. """
    revised_output = "The response is sufficient."
    feedback = ""
    for i in range(max_iterations):
        prompt = user_prompt if i == 0 else f"Feedback: {feedback}, Revised Output: {revised_output}"
        messages = [
            {"role": "system",
            "content": system_prompt
            },
            {"role": "user",
            "content": f"{prompt}"
            }
        ]
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        content = response.choices[0].message.content
        
        # convert to valid json
        valid_json = content
        content_json = json.loads(valid_json)
        
        # extract rating
        rating_str = content_json['rating']
        rating_num = int(re.findall(r'\d+', rating_str)[0])
        
        # extract feedback
        feedback = content_json['feedback']
        
        # extract revised output
        revised_output = content_json['revised_output']
        print(f"i={i}, Rating: {rating_num}, \nFeedback: {feedback}, \nRevised Output: {revised_output}")
        if rating_num >= target_rating:
            return revised_output
        
    return revised_output

#%% Test
user_prompt = "The American Civil War was a civil war in the United States between north and south."
res = self_feedback(user_prompt=user_prompt, max_iterations=3, target_rating=95)
res
# %%
pprint(res)