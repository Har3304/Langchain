!pip install pypdf
import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (TextLoader, PyPDFLoader)
from google.colab import userdata

hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

def load_textfile(file_path: str):
  with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
    temp_file.write(b'Hello This is temporary text file. \n This is used to test the demo for textloader. ')
    tempfile_path = temp_file.name

  try:
    loader = TextLoader(tempfile_path)
    documents = loader.load()

    print(f'Loaded {len(documents)} document(s).')
    print(f'Content Preview: {documents[0].page_content[:100]}...')
    print(f'Meta data: {documents[0].metadata}')
    # for doc in documents:
    #   print(doc)      
    #   print(doc.page_content)
  finally:
    os.remove(tempfile_path)
print(f'---------TextLoader----------')
print(load_textfile('temp.txt'))

def pdf_loader(file_path: str):  
  try:
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f'Loaded {len(documents)} document(s).')
    for i, doc in enumerate(documents):
        print(f'Document {i+1} Content Preview: {doc.page_content[:100]}')
        print(f'Metadata: {doc.metadata}')
    # for doc in documents:
    #   print(doc)      
    #   print(doc.page_content)
  except Exception as e:
    print(f'Error loading PDF: {e}')
  

print(f'---------PDFLoader----------')

pdf_loader('/content/LangChain_Document_Loaders_One_Page_Guide.pdf')

print(f'')
