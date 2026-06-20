!pip install pypdf
import os
import tempfile
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

def load_document(path: str):
  loader = PyPDFLoader(path)
  docs = loader.load()
  return docs

def similarity_scores_with_scores():
  with tempfile.TemporaryDirectory() as tmpdir:    
    vectorstore = Chroma.from_documents(documents=load_document('/content/LangChain_Document_Loaders_One_Page_Guide.pdf'), embedding=embedding_model, persist_directory=tmpdir)

    query = 'How to load text from text files?'
    results_with_scores = vectorstore.similarity_search_with_score(query, k=3)

    print(f'Top 3 results with score for query: {query}: \n')
    for i, (doc, score) in enumerate(results_with_scores):
      print(f"Result {i+1}: {doc.page_content} Score: {score:.4f}, Source: {doc.metadata} ")
similarity_scores_with_scores()
