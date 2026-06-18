# Model that remembers
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

endpoint_llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
    task = 'text-generation',
    temperature = 0.1,
    max_new_tokens = 2000
)
llm = ChatHuggingFace(llm=endpoint_llm)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and witty AI assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")])
chain = prompt | llm

store = {}

def get_session_history(session_id: str):
  if session_id not in store:
    store[session_id] = InMemoryChatMessageHistory()
  return store[session_id]

with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key = 'input',
    history_messages_key = 'history'
)

config ={'configurable': {'session_id':'user_123'}}
response_1 = with_message_history.invoke(
    {'input': 'Hi! My name is Juan and I love coding in python'},
    config = config)
print('AI: ', response_1)
response_2 = with_message_history.invoke(
    {'input':'What is my name and what do I like to do?'},
    config = config)
print('AI: ', response_2)
