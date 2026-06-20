from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from google.colab import userdata
import os

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN']= hf_token

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text_generation',
    temperature = 0.1,
    max_new_tokens = 150)

chatbot = ChatHuggingFace(llm=llm)

prompt = ChatPromptTemplate.from_messages(
    [('system', 'You are a helpful and witty AI assistant'),
    MessagesPlaceholder(variable_name='history'),
    ('human', '{question}')])

chain = prompt | chatbot

def get_session_history(session_id: str):
  return SQLChatMessageHistory(
      session_id = session_id,
      connection = 'sqlite:///chat_history.db')

bot_with_long_term_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='question',
    history_messages_key = 'history')

config_user_1 = {'configurable': {'session_id':'user_abc_123'}}

print('---Turn 1----')
response_1 = bot_with_long_term_memory.invoke(
    {'question' : 'What is the capital of Germany'},
    config = config_user_1)
print(response_1)
response_2 = bot_with_long_term_memory.invoke(
    {'question':'What was the first question asked in this chat session?'},
    config = config_user_1)
print(response_2)
