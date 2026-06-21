from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from google.colab import userdata
import os

# 1. Authenticate with HuggingFace Hub
hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

# 2. Set up the local embedding model and target data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

sample_chunks = [
    Document(page_content="A company code is an independent legal entity in SAP subject to external accounting principles like GAAP or IFRS. All financial transactions are performed here. The unique identifier configuration involves setting a country, currency, and city parameter."),
    Document(page_content="Furthermore, the company code is deeply integrated with other modules. In materials management, goods movements reference it. In sales and distribution, orders and billing documents are linked directly to a specific company code to ensure legal operational consistency.")
]

# Create a clean temporary vector database
db = Chroma.from_documents(sample_chunks, embeddings, collection_name="compression_test")
base_retriever = db.as_retriever(search_kwargs={"k": 2})

# 3. Initialize a fast HuggingFace LLM to act as the Extractor
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.1,
    max_new_tokens=256,
    huggingfacehub_api_token=hf_token
)

llm = ChatHuggingFace(llm=llm)

# 4. Construct the Contextual Compression Retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

# 5. Execute a query to view the compression in action
query = "What are the essential configuration parameters required to create a company code?"
compressed_docs = compression_retriever.invoke(query)

print(f"--- Query: {query} ---\n")
print(f"Number of retrieved documents returned: {len(compressed_docs)}")
for i, doc in enumerate(compressed_docs):
    print(f"\n[Compressed Document {i+1}]")
    print(doc.page_content)
