from langchain_postgres import PGVector
import os
from google.colab import userdata
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DATABASE_URL = userdata.get('SUPABASE_DATABASE_URL')

HF_TOKEN = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HF_TOKEN
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN_READ')


def connect_to_supabase():
  embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
  vectorstore = PGVector(
      embeddings=embeddings,
      collection_name = 'production_docs',
      connection = DATABASE_URL,
      use_jsonb=True)
  return vectorstore


def verify_connection():
  test_doc = Document(
      page_content='This is a test document to verify Supabase connection',
      metadata = {'test':True}
  )
  vectorstore = connect_to_supabase()
  try:
    ids = vectorstore.add_documents([test_doc])
    print(f'Added test document with ID: {ids[0]}')

    results = vectorstore.similarity_search('text document')
    if results:
      print(f'Search Works! : {results[0].page_content}')

    vectorstore.delete(ids)
    print('Cleanup Complete!')

    return True
  except Exception as e:
    print(f'Error: {e}')
    return False

verify_connection()
