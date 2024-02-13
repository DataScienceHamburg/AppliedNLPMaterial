#%% packages
from transformers import pipeline
import pandas as pd

#%% Classifier
task = "summarization"
model = "sshleifer/distilbart-cnn-12-6"
summarizer = pipeline(task= task, model=model)

# %% Data Preparation
# ChatGPT prompt:  "Can you please create some text which i will use for text summarization with huggingface? It should include a programmer and an AI."
story = """
John, a skilled programmer, was facing a daunting challenge. He had been tasked with developing a cutting-edge AI system that could revolutionize the way businesses handle customer inquiries. With the rise of chatbots and virtual assistants, there was immense pressure on John to create an AI that could not only understand and respond to natural language queries but also learn and adapt over time.

John's journey began with researching state-of-the-art natural language processing models. He dove into the world of machine learning and neural networks, spending countless hours studying algorithms and data structures. As he delved deeper into the intricacies of AI, he became increasingly fascinated by the potential of his creation.

After months of tireless work, John's AI system started to take shape. He integrated the latest transformer models and fine-tuned them with vast datasets of customer interactions. The AI began to demonstrate remarkable abilities, understanding context, and providing nuanced responses. It could even detect emotions and tailor its replies accordingly.

As the AI continued to evolve, John marveled at its progress. It had become more than just a piece of code; it was a digital collaborator, a partner in problem-solving. Together, they were poised to change the landscape of customer service, making it more efficient and user-friendly.

However, John was aware of the ethical considerations surrounding AI. He implemented robust safeguards to ensure the system remained fair, unbiased, and respectful of user privacy. He knew that with great power came great responsibility.

In the end, John's AI project was a testament to the incredible synergy between human creativity and artificial intelligence. It was a story of determination, innovation, and the endless possibilities that emerge when a programmer's skills meet the capabilities of AI.
"""
# %%
result = summarizer(story, min_length=20, max_length=80, do_sample=False)
result[0]['summary_text']
# %% number of characters
len(result[0]['summary_text'].split(' '))


# %%
