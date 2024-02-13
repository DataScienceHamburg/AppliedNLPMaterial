#%% packages
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from pprint import pprint

#%%
# download the model from HuggingFace
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
# save in subfolder "models"

# %%
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q3_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

#%% system and user prompt
system_prompt = "Eve lives in Hamburg.; Bob lives in Cape Town.; Alice lives in Mumbay."
user_prompt = "Where does Eve live?"
#%% naive approach
prompt_naive = f"{system_prompt}\n{user_prompt}"

llm(prompt_naive)
#%% set up prompt correctly
llama_prompt = f"<s>[INST]<<SYS>>\n{system_prompt}<</SYS>>\n{user_prompt}[/INST]"
pprint(llama_prompt)
#%% run Llama2
llm(llama_prompt)
