#%%
import anthropic
import os
#%%
client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key= os.environ.get("CLAUDE_API"),
)
#%%
message = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "You want to build a team of 11 players for soccer. All players are animals. Which animals in the different positions yield the best performance in a tournament?"}
    ]
)
print(message.content)
#%%
from pprint import pprint
pprint(message.content[0].text)
# %%
