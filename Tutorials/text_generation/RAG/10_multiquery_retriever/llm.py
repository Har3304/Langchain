import logging
import sys

# Update the logger name to match 'langchain_classic'
logger = logging.getLogger('langchain_classic.retrievers.multi_query')
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
logger.addHandler(handler)



from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from google.colab import userdata
import os

# 2. Authenticate Hugging Face
hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token

# 3. Define the LLM (Using the corrected Meta-Llama-3 repo ID)
model_for_chat = HuggingFaceEndpoint(
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct',
    temperature=0.1,
    max_new_tokens=300
)
llm = ChatHuggingFace(llm=model_for_chat)

# 4. Define the Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 5. Define Documents
TECH_DOCS = [
    Document(
        page_content="The company's new AI strategy focuses on ethical development and deployment of machine learning models across all product lines. This includes strict data privacy guidelines and bias detection mechanisms.",
        metadata={"doc_id": "AI-001", "category": "Strategy", "version": "1.0"}
    ),
    Document(
        page_content="Our cloud infrastructure migration project is scheduled for Q3. Key components include moving all legacy systems to AWS S3 and implementing a serverless architecture for new services.",
        metadata={"doc_id": "CLOUD-002", "category": "Infrastructure", "phase": "Planning"}
    ),
    Document(
        page_content="Frontend development guidelines mandate the use of React 18, TypeScript, and a component-based design system to ensure consistency and maintainability of the user interface.",
        metadata={"doc_id": "FRONTEND-003", "category": "Development", "tech": "React"}
    ),
    Document(
        page_content="The cybersecurity protocol requires all employees to undergo annual training, use multi-factor authentication, and report suspicious activities immediately to the IT security team.",
        metadata={"doc_id": "SECURITY-004", "category": "Compliance", "frequency": "Annual"}
    ),
    Document(
        page_content="Backend services are built using Python with FastAPI for RESTful APIs and PostgreSQL for database management. Docker containers are used for deployment across Kubernetes clusters.",
        metadata={"doc_id": "BACKEND-005", "category": "Development", "tech": "Python"}
    ),
    Document(
        page_content="Project management methodology follows Agile scrum principles, with bi-weekly sprints, daily stand-ups, and regular sprint reviews to ensure iterative development and continuous feedback.",
        metadata={"doc_id": "PROJECT-006", "category": "Process", "methodology": "Agile"}
    ),
    Document(
        page_content="Data analytics best practices involve using Apache Spark for large-scale data processing, Google BigQuery for warehousing, and Tableau for interactive data visualization and reporting.",
        metadata={"doc_id": "DATA-007", "category": "Analytics", "tools": "Spark"}
    ),
    Document(
        page_content="Our mobile application development uses native iOS (Swift) and Android (Kotlin) for optimal performance, complemented by Firebase for backend services and push notifications.",
        metadata={"doc_id": "MOBILE-008", "category": "Development", "platform": "Mobile"}
    ),
    Document(
        page_content="Quality assurance procedures include automated testing with Selenium, manual testing for critical paths, and user acceptance testing (UAT) before every major release.",
        metadata={"doc_id": "QA-009", "category": "Testing", "type": "Automated"}
    ),
    Document(
        page_content="The customer support system integrates Zendesk for ticket management, a knowledge base for self-service, and AI chatbots for immediate query resolution and escalation.",
        metadata={"doc_id": "SUPPORT-010", "category": "Operations", "system": "Zendesk"}
    ),
    Document(
        page_content="Research and development initiatives explore quantum computing applications for cryptography and advanced materials science, with collaborations planned with leading universities.",
        metadata={"doc_id": "R&D-011", "category": "Innovation", "focus": "Quantum"}
    )
]

def create_base_vectorstore():
    return Chroma.from_documents(documents=TECH_DOCS, embedding=embeddings)

def demo_multiueryretriever():
    print('*'*30)
    print('*'*5 + ' MULTIQUERY-RETRIEVER ' + '*'*5)
    print('*'*30)

    vectorstore = create_base_vectorstore()

    # Initialize MultiQueryRetriever
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={'k': 2})
    )

    tech_query = "What are the cybersecurity protocols mentioned in the documents?"
    print(f'Query: {tech_query}\n')
    print('--- Triggering MultiQuery Retrieval (Watch for the generated queries below) ---')

    # Execution
    docs = retriever.invoke(tech_query)

    print('\n--- Results ---')
    print(f'Retrieved {len(docs)} unique documents.')
    for i, doc in enumerate(docs):
        # Fixed: Changed 'topic' to 'category' to match your Document metadata keys
        category = doc.metadata.get('category', 'N/A')
        print(f"Document {i+1}. [{category}] {doc.page_content[:100]}...")
    return docs
# Run the function
docs = demo_multiueryretriever()
print(docs)

vectorstore=create_base_vectorstore()
retriever = MultiQueryRetriever.from_llm(llm=llm, retriever = vectorstore.as_retriever(search_kwarg={'k':2}))

def format_doc_with_source(docs):
  formatted=[]
  for i, doc in enumerate(docs):
    source = doc.metadata.get('source', 'unknown')
    formatted.append(f"[{i+1}] {source}: \n {doc.page_content}")
  return "\n\n".join(formatted)


prompt = """Answer the question based on given context below. Include which source you used.
            context: {context}
            question: {question}
            Answer (include source): """


prompt = PromptTemplate(template=prompt)
rag_chain = (
    {
      "context": retriever | format_doc_with_source, 
      'question':RunnablePassthrough()
     }
    | prompt
    | llm
    | StrOutputParser())

print('*'*30)
print('*'*2 + ' MULTIQUERY-RETRIEVER LLM ' + '*'*2)
print('*'*30)

query = "What are the cybersecurity protocols mentioned in the documents?"
print("\n--- Executing RAG Chain ---")
response = rag_chain.invoke(query)
print(response)
