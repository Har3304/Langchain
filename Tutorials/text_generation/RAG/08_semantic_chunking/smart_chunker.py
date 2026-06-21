import numpy as np
import re
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

class HybridFallbackChunker:
    def __init__(self, embedding_model, max_chunk_size: int = 600, similarity_threshold: float = 0.75):
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
        embeddings = self.embedding_model.embed_documents(sentences)
        semantic_groups = []
        current_group = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i-1], embeddings[i])
            
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


################### USECASE ######################

# Initialize with a strict structural maximum capacity

TEXT_TO_SPLIT = """A company code is a fundamental organizational unit within various enterprise resource planning (ERP) systems, particularly SAP. It represents an independent legal entity that is subject to external accounting principles, such as national GAAP (Generally Accepted Accounting Principles) or IFRS (International Financial Reporting Standards). All financial transactions and reporting in an organization are typically performed at the company code level.

Each company code operates as a self-contained unit for external accounting purposes. This means it can generate its own balance sheets and profit and loss statements. It plays a crucial role in defining the scope of financial reporting, as legal requirements often mandate financial statements for each distinct legal entity within a corporate group. The creation of a company code involves defining its unique identifier, currency, country, and city, among other essential parameters.

Furthermore, the company code is integrated with other modules within an ERP system. For instance, in materials management, goods movements are often posted with reference to a company code. In sales and distribution, sales orders and billing documents are linked to a specific company code. This integration ensures consistency and accuracy across all business processes and provides a unified view of financial data. Understanding the structure and function of company codes is essential for anyone working with ERP systems in a financial or operational capacity.

Finally, maintaining proper governance and data integrity for company codes is paramount. Changes to company code configurations can have wide-ranging impacts on financial reporting, taxation, and business operations. Therefore, robust change management processes and regular audits are necessary to ensure compliance and prevent data discrepancies. Best practices also suggest clear documentation of each company code's purpose and its relationship to other organizational units within the enterprise."""


smart_fallback_chunker = HybridFallbackFallbackChunker(
    embedding_model=embeddings,
    max_chunk_size=550,          # The hard target limit 
    similarity_threshold=0.75     # Primary semantic boundary
)

chunks = smart_fallback_chunker.split_text(TEXT_TO_SPLIT)

print(f"Total processed chunks: {len(chunks)}")
for i, chunk in enumerate(chunks):
    print(f"\n[Chunk {i+1} | Length: {len(chunk.page_content)} characters]")
    print(chunk.page_content)
