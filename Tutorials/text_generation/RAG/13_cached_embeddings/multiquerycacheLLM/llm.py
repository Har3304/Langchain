import hashlib
import numpy as np
import os
from google.colab import userdata


from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManager


token = userdata.get('HF_TOKEN_READ') or userdata.get('HF_TOKEN')
if token:
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = token
    os.environ['HF_TOKEN'] = token


class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):  
        self.cache = {}        
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = similarity_threshold
  
    def _cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get(self, query: str):
        if not self.cache:
            return None
                    
        query_embedding = self.embeddings.embed_query(query)        
        best_match = None
        highest_similarity = -1.0                
        for cache_id, cached_item in self.cache.items():
            similarity = self._cosine_similarity(query_embedding, cached_item["embedding"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = cached_item                        
        if highest_similarity >= self.similarity_threshold:
            return best_match            
        return None

    def set_(self, query: str, response: str):        
        normalized = query.lower().strip()
        query_hash = hashlib.md5(normalized.encode()).hexdigest()                
        query_embedding = self.embeddings.embed_query(query)        
        self.cache[query_hash] = {
            "query": query,
            "response": response,
            "embedding": query_embedding
        }

    def stats(self) -> dict:
        return {"cached_queries": len(self.cache)}


class MultiQueryCacheLLM:
    def __init__(self, similarity_threshold: float = 0.95):
        self.llm = HuggingFaceEndpoint(
            repo_id='meta-llama/Llama-3.1-8B',
            task='text-generation',
            temperature=0.1,
            max_new_tokens=200)
        self.cache = SemanticCache(similarity_threshold=similarity_threshold)
        self.cache_hits = 0
        self.cache_miss = 0

    def mult_queries(self, query: str):      
        retriever = MultiQueryRetriever.from_llm(retriever=Chroma().as_retriever(), llm=self.llm)
        run_manager = CallbackManager(handlers=[]).on_retriever_start(serialized={}, query=query)
        generated_queries = retriever.generate_queries(query, run_manager=run_manager)    
        return generated_queries, query

    def invoke(self, query: str):
        multi_queries, original_query = self.mult_queries(query)        
        all_queries = list(set([original_query] + multi_queries))        
        for q in all_queries:    
            cached_response = self.cache.get(q)
            if cached_response:
                self.cache_hits += 1
                return cached_response['response'], True                
        self.cache_miss += 1
        response = self.llm.invoke(original_query)
        
        
        self.cache.set_(original_query, response)
        return response, False
  
    def stats(self):
        total = self.cache_hits + self.cache_miss
        return {
            "cache_hits": self.cache_hits,
            "cache_miss": self.cache_miss,
            "cache_hit_ratio": self.cache_hits / total if total > 0 else 0
        }
