import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from google.colab import userdata

# Fetch token securely from Colab Secrets
hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

# Model Configuration
repo_id = 'meta-llama/Llama-3.1-8b-Instruct'

def q_a(query):
    # Define prompt template
    prompt = PromptTemplate(
        template="""You are a helpful and witty AI assistant. 
        Answer the question asked by user: {query}""",
        input_variables=['query']
    )
    
    # Initialize the Hugging Face Inference Endpoint
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=200
    )
    
    # Wrap it in a Chat-Model layer for instruction tracking
    bot = ChatHuggingFace(llm=llm)
    chain = prompt | bot
    
    response = chain.invoke({'query': query})
    return response.content

# Continuous loop for interacting with the bot
while True:
    user_input = input('Ask question (q to exit): ')
    if user_input.lower() == 'q':
        print('bye')
        break
    print(q_a(user_input))
