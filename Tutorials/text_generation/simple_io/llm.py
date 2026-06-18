# This code is exclusively for google colab
import os
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from google.colab import userdata

# The code requires huggingface read access token and saved in secrets on google colab

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
repo_id = 'meta-llama/Llama-3.1-8b-Instruct'

def q_a(query):
  prompt = PromptTemplate(
                  template = f"""You are a helpful and witty AI assistant. 
                  Anser the question asked by user: {query}"""
                  input_variable = ['query'])
  llm = HuggingFaceEndpoint(repo_id = repo_id,
                            task = 'text-generation',
                            temperature = 0.1,
                            max_new_tokens = 200)
  bot = ChatHuggingFace(llm=llm)
  chain = prompt | bot
  response = chain.invoke({'query':query})
  return response
while True:
  user_input = input('Ask question (q to exit): ')
  if user_input.lower() == 'q':
    print('bye')
    break
  print(q_a(user_input))
