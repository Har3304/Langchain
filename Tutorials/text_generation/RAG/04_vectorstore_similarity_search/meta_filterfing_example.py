import tempfile
from langchain_core.documents import Document

def metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        
        docs = load_document('/content/LangChain_Document_Loaders_One_Page_Guide.pdf')
        
        
        for doc in docs:
            doc.metadata['topic'] = 'langchain'
            
        
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=tmpdir
        )
        
        
        filter_criteria = {'topic': 'langchain'}    
        query = "How to load pdf files?"
        results = vectorstore.similarity_search(query, filter=filter_criteria, k=1)
        
        return results

metadata_filtering()
