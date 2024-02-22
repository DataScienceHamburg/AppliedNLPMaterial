#%% packages
from transformers import pipeline


#%% Text Summarization
pipe = pipeline("summarization", model="Falconsai/text_summarization")

article = """The company was founded in 2016 by French entrepreneurs Clément Delangue, Julien Chaumond, and Thomas Wolf in New York City, originally as a company that developed a chatbot app targeted at teenagers.[1] The company was named after the "hugging face" emoji.[1] After open sourcing the model behind the chatbot, the company pivoted to focus on being a platform for machine learning.

In March 2021, Hugging Face raised US$40 million in a Series B funding round.[2]

On April 28, 2021, the company launched the BigScience Research Workshop in collaboration with several other research groups to release an open large language model.[3] In 2022, the workshop concluded with the announcement of BLOOM, a multilingual large language model with 176 billion parameters.[4][5]

In December 2022, the company acquired Gradio, an open source library built for developing machine learning applications in Python.[6]

On May 5, 2022, the company announced its Series C funding round led by Coatue and Sequoia.[7] The company received a $2 billion valuation.

On August 3, 2022, the company announced the Private Hub, an enterprise version of its public Hugging Face Hub that supports SaaS or on-premises deployment.[8]

In February 2023, the company announced partnership with Amazon Web Services (AWS) which would allow Hugging Face's products available to AWS customers to use them as the building blocks for their custom applications. The company also said the next generation of BLOOM will be run on Trainium, a proprietary machine learning chip created by AWS.[9][10][11]

In August 2023, the company announced that it raised $235 million in a Series D funding, at a $4.5 billion valuation. The funding was led by Salesforce, and notable participation came from Google, Amazon, Nvidia, AMD, Intel, IBM, and Qualcomm.[12]
"""
pipe(article, min_length=5, max_length=20)

# %%
