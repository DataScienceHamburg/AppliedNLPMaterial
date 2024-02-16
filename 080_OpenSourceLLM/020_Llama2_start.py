#%% packages
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

#%%
# download the model from HuggingFace
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
# save in subfolder "models"

# %% Callback Manager
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

#%% set up prompt correctly

#%% run Llama2
