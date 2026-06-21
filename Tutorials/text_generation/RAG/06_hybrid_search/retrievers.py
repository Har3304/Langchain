# Hybrid search
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from google.colab import userdata
from langchain_core.documents import Document # Ensure Document is imported
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever # Corrected import path for EnsembleRetriever

HF_TOKEN = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN

documents = [
    Document(page_content="A bunch of scientists bring back dinosaurs and mayhem breaks loose", metadata={"source": "movie_summary_1"}),
    Document(page_content="Leo DiCaprio gets lost in a dream within a dream within a dream within a ...", metadata={"source": "movie_summary_2"}),
    Document(page_content="A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea", metadata={"source": "movie_summary_3"}),
    Document(page_content="A bunch of normal-sized women are supremely wholesome and some men pine after them", metadata={"source": "movie_summary_4"}),
    Document(page_content="Three men walk into the Zone, three men walk out of the Zone, the Zone is empty", metadata={"source": "movie_summary_5"})
]

embedding = HuggingFaceEmbeddings()
vectorstore = Chroma.from_documents(documents=documents, embedding=embedding, collection_name='hybrid_test')

# retriever
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 3})
print('Vectorstore ready: ')

# BM25Retriever
bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3)
print('BM25 Retriever ready: ')

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(retrievers=[retriever, bm25_retriever], weights=[0.5, 0.5])
print('Hybrid Retriever ready: ')

queries = [
    "What movie is about dinosaurs?",
    "Tell me about a movie with dreams within dreams.",
    "Which movie features a psychologist detective?",
    "What is the summary of a movie about normal women?",
    "Give me information about a movie where three men enter and leave the Zone."
]

def test_queries(query, name, retriever):
  results = retriever.invoke(query)
  print(f'\\n {name}- Query: \"{query}\"')
  for i, doc in enumerate(results[:3]):
    preview = doc.page_content[:80] + '...'
    print(f' {i+1}. {preview}')
  return results

for q in queries:
  print(test_queries(q, 'BM25', bm25_retriever))
  print(test_queries(q, 'VectorEmbedding Retriever', retriever))
  print(test_queries(q, 'Hybrid', ensemble_retriever))
  print('--------\n\n-----------')
  print('--------\n\n-----------')

def hybrid_retriever(query, retrievers, weights, k-3, rrf_k=60):
  doc_scores = {}
  for retriever, weight in zip(retrievers, weights):
    results = retriever.invoke(query)
    for rank, doc in enumerate(results):
      key = doc.page_content
      rrf_score = weight * (1.0 / (rank + 1))
      if key in doc_scores:
        doc_scores[key] = doc_scores[key][0] + rrf_score
      else:
        doc_scores[key] = (rrf_score, doc)

  sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
  return [doc for _, doc in sorted_docs[:k]]
