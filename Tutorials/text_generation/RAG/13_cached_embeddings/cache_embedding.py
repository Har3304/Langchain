from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
import numpy as np
import tempfile
from google.colab import userdata
import os

token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = token
os.environ['HF_TOKEN'] = token

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

def embedding_cache():
  with tempfile.TemporaryDirectory() as dif:
    file_store = LocalFileStore(dif)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=file_store,
        namespace='exercise')    
    texts = [
        "Hello, How are ya?",
        "This is a temporary text line list",
        "For experimental purpose only."]
    
    print(f'First call (not cached) hits the API:')
    vectores1 = cached_embeddings.embed_documents(texts)
    print(f'Embedded document: {len(vectores1)} documents.')

    print(f'Second call (cached) hits the API:')
    vectores2 = cached_embeddings.embed_documents(texts)
    print(f'Embedded document: {len(vectores2)} documents.')
    
    print(f'\n Same vectors: {np.allclose(vectores1[0], vectores2[0])}')

embedding_cache()
