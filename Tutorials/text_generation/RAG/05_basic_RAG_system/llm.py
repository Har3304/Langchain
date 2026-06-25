import os
from google.colab import userdata
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
import random

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
os.environ['HF_TOKEN'] = hf_token

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
session_history = {}
def get_response_from_doc(path:str, query:str):
  loader = PyPDFLoader(path)
  documents = loader.load()

  vectorstore = Chroma()

  for i, doc in enumerate(documents):
    vectorstore.add_documents(documents = [doc], ids = str(i))

  retriever = vectorstore.as_retriever(search_kwargs = {'k': 2})

  context = retriever.invoke(query)  
  context_text = "\n\n".join([doc.page_content for doc in context])

  llm = HuggingFaceEndpoint(
      repo_id = 'meta-llama/Meta-Llama-3-8B-Instruct',
      task = 'text-generation',
      temperature = 0.1,
      max_new_tokens = 1000)

  bot = ChatHuggingFace(llm=llm)

  prompt = ChatPromptTemplate.from_messages([
      ('system', 'You are a helpful and witty AI assistant. Answer user question from given context: {context_text}.'),
      MessagesPlaceholder(variable_name = 'history'),
      ('human', '{question}')
  ])

  
  def get_session_history(session_id: str):
    if session_id not in session_history:
      session_history[session_id] = InMemoryChatMessageHistory()
    return session_history[session_id]

  config = {'configurable': {'session_id':'user_123'}}

  chain = prompt | bot
  with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key = 'history')
  
  response = with_message_history.invoke({'question':query, 'context_text': context_text}, config=config)
  return response.content

print(get_response_from_doc('/content/Class_Valuation_Interview_Prep.pdf', 'How does ClassValuation work?', session_id = ))
print(get_response_from_doc('/content/Class_Valuation_Interview_Prep.pdf', 'What was my first question?'))
