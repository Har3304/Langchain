from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector, PGEngine, PGVectorStore
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.colab import userdata
from langchain_core.documents import Document
import numpy as np
from typing import List
import os
import re


hf_token = userdata.get('HF_TOKEN_READ')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
os.environ['HF_TOKEN'] = hf_token
supabase_url = userdata.get('SUPABASE_DATABASE_URL')

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Extract your base URL (assuming it starts with postgresql:// or postgres://)
base_url = userdata.get('SUPABASE_DATABASE_URL')

# Ensure the driver is explicitly set to use psycopg (v3 compatible with async)
if base_url.startswith("postgresql://"):
    supabase_url = base_url.replace("postgresql://", "postgresql+psycopg://", 1)
elif base_url.startswith("postgres://"):
    supabase_url = base_url.replace("postgres://", "postgresql+psycopg://", 1)
else:
    supabase_url = base_url

# Initialize the connection engine with the correct driver
connection_engine = PGEngine.from_connection_string(
    url=supabase_url,  # Explicitly using +psycopg
    pool_size=3,
    max_overflow=0
)

class HybridFallbackChunker:
    def __init__(self, embedding_model=embeddings, max_chunk_size: int = 600, similarity_threshold: float = 0.75):
        """
        Employs Semantic chunking as the primary segmentation strategy.
        If any semantic chunk violates structural size constraints, it runs a
        Recursive Character Splitter as an automated fallback.
        """
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold

        # Initialize the secondary structural fallback engine
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=int(max_chunk_size * 0.15),  # 15% standard semantic safety overlap
            separators=['\n\n', '\n', ' ', '']
        )

    def _split_into_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s]

    def _cosine_similarity(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def split_text(self, text: str) -> List[Document]:
        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        # 1. Primary Strategy: Build Semantic Groups
        embeddings_list = self.embedding_model.embed_documents(sentences)
        semantic_groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings_list[i-1], embeddings_list[i])

            if similarity >= self.similarity_threshold:
                current_group.append(sentences[i])
            else:
                semantic_groups.append(" ".join(current_group))
                current_group = [sentences[i]]

        if current_group:
            semantic_groups.append(" ".join(current_group))

        # 2. Secondary Strategy: Enforce Over-Size Fallback
        final_documents = []
        for raw_chunk in semantic_groups:
            if len(raw_chunk) <= self.max_chunk_size:
                # Primary victory: the semantic chunk fits within constraints
                final_documents.append(Document(page_content=raw_chunk))
            else:
                # Fallback triggered: programmatic splitting of oversized semantic blocks
                sub_chunks = self.fallback_splitter.split_text(raw_chunk)
                for sub in sub_chunks:
                    final_documents.append(Document(page_content=sub))

        return final_documents

class PGVectorLongtermChatBot(HybridFallbackChunker):
    def __init__(self, collection_name, connection_engine, embeddings, repo_id):
        super().__init__(embedding_model=embeddings)

        self.embeddings = embeddings
        self.collection_name = collection_name


        self.vectorstore = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=supabase_url)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3})

        self.repo_id = repo_id
        self.llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            task='text-generation',
            temperature=0.1,
            max_new_tokens=300
        )
        self.bot = ChatHuggingFace(llm=self.llm)

    def get_session_history(self, session_id: str):
        # This helper returns the SQL-backed history for a given session
        return SQLChatMessageHistory(
            session_id=session_id,
            connection='sqlite:///chat_history.db'
        )

    def retrieve_document(self, query):
        retrieved_docs = self.retriever.invoke(query)
        context_strings = [doc.page_content for doc in retrieved_docs]
        return '\n\n'.join(context_strings)

    def ask(self, query, session_id="default_session"):
        self.context = self.retrieve_document(query)
        system_prompt = (
            "You are a strict QA assistant. Your ONLY job is to answer the user's question "
            "using exclusively the provided text context below. \n\n"
            f"CRITICAL CONTEXT:\n{self.context}\n\n"
            "RULES:\n"
            "1. You must only use facts directly mentioned in the context.\n"
            "2. If the answer cannot be found in the context, you must reply exactly with: "
            "'I am sorry, but that information is not present in the provided document.'\n"
            "3. Do NOT make up information or use any outside knowledge."
        )

        prompt = ChatPromptTemplate.from_messages([
            ('system', system_prompt),
            MessagesPlaceholder(variable_name='chat_history'),
            ('user', '{query}')
        ])

        chain = prompt | self.bot

        with_messages_history = RunnableWithMessageHistory(
            chain,
            self.get_session_history,          # Pass the method reference, don't execute it here
            input_messages_key='query',
            history_messages_key='chat_history' # Must match the prompt placeholder
        )

        response = with_messages_history.invoke(
            {"query": query},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

while True:
  bot = PGVectorLongtermChatBot(
    collection_name='Class_Valuation_Interview_Prep',
    connection_engine=connection_engine,
    embeddings=embeddings,
    repo_id='meta-llama/Meta-Llama-3-8B-Instruct')

  query = str(input('Ask question (about ClassValuation or "q" to exit):'))
  if query.lower() == 'q':
    print('Good Bye!')
    break
  print(bot.ask(query))
