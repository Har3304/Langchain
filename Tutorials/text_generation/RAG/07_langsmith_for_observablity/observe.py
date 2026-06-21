import os
from google.colab import userdata
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langsmith import traceable, Client

# 1. Grab tokens from Colab secrets
tf_token = userdata.get('HF_TOKEN_READ')
langsmit_api = userdata.get('LANGSMITH_API')

# 2. Set BOTH modern and legacy variables to ensure complete coverage
os.environ['HUGGINGFACEHUB_API_TOKEN'] = tf_token

# Modern LangSmith variables
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_API_KEY'] = langsmit_api
os.environ['LANGSMITH_PROJECT'] = 'Colab_Tracing_Demo'
os.environ['LANGSMITH_ENDPOINT'] = 'https://api.smith.langchain.com'

# Legacy LangChain variables (Fallback)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_API_KEY'] = langsmit_api
os.environ['LANGCHAIN_PROJECT'] = 'Colab_Tracing_Demo'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'


# 3. Explicitly set the target project name inside the decorator
@traceable(
    name='basic_chain', 
    tags=['production', 'summarization'],
    project_name='Colab_Tracing_Demo'
)
def demo_basic_tracing():
    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Llama-3.1-8B-Instruct',
        task='text-generation',
        temperature=0.2,
        max_new_tokens=200
    )
    chatbot = ChatHuggingFace(llm=llm)
    prompt = ChatPromptTemplate.from_template(
        template='Explain topic in one sentence: {topic}',
        output_parser=StrOutputParser()
    )
  
    chain = prompt | chatbot | StrOutputParser()
    response = chain.invoke({'topic': 'dogs'})
    print(response)

# Run the tracing function
demo_basic_tracing()

# 4. CRITICAL: Force clear the asynchronous API delivery queues before Colab cuts the runtime connection
print("\n[LangSmith] Explicitly flushing tracing queues...")
client = Client()
