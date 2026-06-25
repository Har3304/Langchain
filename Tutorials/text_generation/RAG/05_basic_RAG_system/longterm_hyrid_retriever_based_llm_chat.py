import os
from google.colab import userdata
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.retrievers import BM25Retriever  
from langchain_classic.retrievers import EnsembleRetriever


hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HF_TOKEN'] = hf_token

loader = PyPDFLoader('/content/Class_Valuation_Interview_Prep.pdf')
documents = loader.load()
b25 = BM25Retriever.from_documents(documents=documents, k=2)

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = Chroma(embedding_function=embeddings)
vectorstore.add_documents(documents=documents, ids=[str(i) for i in range(len(documents))])
vector_retriever = vectorstore.as_retriever(search_kwargs={'k': 2})


hybrid_retriever = EnsembleRetriever(retrievers=[b25, vector_retriever], weights=[0.5, 0.5])


query = 'How does ClassValuation work?'
results = hybrid_retriever.invoke(query)
context_str = '\n\n'.join([doc.page_content for doc in results])

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a helpful and witty AI assistant. From the given context answer user question.'),
    MessagesPlaceholder(variable_name='history'),
    ('human', 'question: {question}\ncontext: {context}')
])


def get_session_id(session_id: str):
    return SQLChatMessageHistory(
        session_id=session_id,
        connection='sqlite:///chat_history.db'
    )


llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    task='text-generation',
    temperature=0.1,
    max_new_tokens=1000
)
bot = ChatHuggingFace(llm=llm)
chain = prompt | bot


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_id,
    input_messages_key='question',   
    history_messages_key='history'    
)


config = {'configurable': {'session_id': 'user_134'}}
result = with_message_history.invoke(
    input={'question': query, 'context': context_str}, 
    config=config
)

print(result.content)
