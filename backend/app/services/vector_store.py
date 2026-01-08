"""
Vector store service using ChromaDB for embeddings storage and retrieval.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_model: str = "text-embedding-3-large"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize OpenAI client using environment variables
        self.openai_client = OpenAI()
        
        # Initialize ChromaDB
        self._initialize_chroma()
    
    def _initialize_chroma(self):
        """Initialize ChromaDB client and collection."""
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB collection '{self.collection_name}'")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts using OpenAI."""
        loop = asyncio.get_event_loop()
        
        def _get_embeddings():
            try:
                logger.info(f"Generating embeddings for {len(texts)} texts using {self.embedding_model}")
                response = self.openai_client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                embeddings = [embedding.embedding for embedding in response.data]
                logger.info(f"Successfully generated {len(embeddings)} embeddings")
                return embeddings
            except Exception as e:
                logger.error(f"Failed to generate embeddings: {e}")
                raise
        
        return await loop.run_in_executor(self.executor, _get_embeddings)
    
    async def add_documents(
        self, 
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """
        Add document chunks to vector store.
        """
        try:
            # Extract texts and metadata
            texts = [chunk["content"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            ids = [chunk["metadata"]["chunk_id"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = await self.generate_embeddings(texts)
            
            # Add to ChromaDB
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.collection.add(
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    ids=ids
                )
            )
            
            logger.info(f"Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            return False
    
    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search for relevant chunks.
        """
        try:
            # Generate query embedding
            query_embeddings = await self.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Perform search
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=filter_metadata
                )
            )
            
            # Process results
            retrieved_chunks = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    similarity_score = 1 - results["distances"][0][i]  # Convert distance to similarity
                    
                    # Apply similarity threshold
                    if similarity_score >= similarity_threshold:
                        chunk_data = {
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i],
                            "similarity_score": similarity_score,
                            "chunk_id": results["ids"][0][i]
                        }
                        retrieved_chunks.append(chunk_data)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks for query")
            return retrieved_chunks
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific document."""
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                lambda: self.collection.get(
                    where={"document_id": document_id}
                )
            )
            
            chunks = []
            if results["documents"]:
                for i in range(len(results["documents"])):
                    chunk_data = {
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "chunk_id": results["ids"][i]
                    }
                    chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {e}")
            return []
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a specific document."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                self.executor,
                lambda: self.collection.delete(
                    where={"document_id": document_id}
                )
            )
            
            logger.info(f"Deleted document {document_id} from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)