import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

print('-----Starting phase 1: Indexing------')

raw_documents = [Document(page_content="Project Alpha's total budget for Q3 2026 is $1,250,000. It is heavily allocated toward research and development."), 
                 Document(page_content="The lead engineer for Project Alpha is Dr. Aris Thorne. Project Beta is led by Sarah Jenkins.")]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
chunks = text_splitter.split_documents(raw_documents)

embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)

print(f"Successfully indexed data into {len(chunks)} searchable chunks.\n")

print('-----Starting phase 2: Querying------')

retriever = vectorstore.as_retriever(search_kwargs={'k':2}) # Changed k from 1 to 2

query = "Who is the lead engineer for project Alpha and how much is the Q3 budget?"

retrieved_doc = retriever.invoke(query)

print(f"User Query: '{query}'")
print(f"Retrieved Context: '{retrieved_doc[0].page_content}'\n")

prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful project assistant. Answer the question based only on the provided context below.
    If you do not know the answer, say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    Answer:""")

def format_docs(docs):
  return '\n\n'.join(doc.page_content for doc in docs)

llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id = 'meta-llama/Llama-3.1-8B-Instruct',
        task = 'conversational',
        temperature = 0.1,
        max_new_tokens = 512,
        huggingfacehub_api_token = hf_token)
)

rag_chain = ({
    'context': retriever | format_docs,
    'question': RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser())

response = rag_chain.invoke(query)
print(f"Generated Response: {response}")
