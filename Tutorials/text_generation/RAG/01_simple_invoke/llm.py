from google.colab import userdata
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os

os.environ['HUGGINGFACEHUB_API_KEY'] = userdata.get('HF_TOKEN_READ')

def main():
  llm = ChatHuggingFace(
      llm=HuggingFaceEndpoint(repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
      task = 'text-generation',
      temperature = 0.9,
      max_new_tokens = 100))
  response = llm.invoke('Say hello to my little friend!')
  return response.content

print(main())
