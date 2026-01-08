"""
Integration tests for the complete RAG pipeline.
"""

import pytest
import tempfile
import os
from unittest.mock import AsyncMock, patch

from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.rag_service import RAGService


@pytest.fixture
async def rag_pipeline():
    """Create a complete RAG pipeline for testing."""
    # Mock components to avoid API calls
    with patch('app.services.vector_store.OpenAI'), \
         patch('app.services.rag_service.OpenAI'), \
         patch('chromadb.PersistentClient'):
        
        # Create services
        doc_processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        
        # Create mock vector store with proper async methods
        vector_store = AsyncMock()
        vector_store.add_documents = AsyncMock(return_value=True)
        vector_store.similarity_search = AsyncMock(return_value=[
            {
                "content": "Machine learning is a subset of artificial intelligence.",
                "metadata": {"filename": "test.txt", "chunk_id": "test_0"},
                "similarity_score": 0.9,
                "chunk_id": "test_0"
            }
        ])
        # get_collection_stats is NOT async - it returns a dict directly
        vector_store.get_collection_stats = lambda: {
            "total_chunks": 5,
            "unique_documents": 1
        }
        
        # Create RAG service with mocked dependencies
        rag_service = RAGService(
            vector_store=vector_store,
            openai_api_key="test_key",
            anthropic_api_key="test_key"
        )
        
        # Mock conversation storage
        rag_service.conversations = {}
        
        # Mock the LLM response methods
        async def mock_generate_openai_response(*args, **kwargs):
            return (
                "Machine learning is indeed a subset of artificial intelligence that enables computers to learn from data.",
                "Generated using test model"
            )
        
        rag_service._generate_openai_response = mock_generate_openai_response
        
        return doc_processor, vector_store, rag_service


class TestRAGPipeline:
    """Integration tests for the complete RAG pipeline."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline(self, rag_pipeline):
        """Test the complete document processing and querying pipeline."""
        doc_processor, vector_store, rag_service = await rag_pipeline
        
        # Create a test document
        test_content = """
        Machine Learning Fundamentals
        
        Machine learning is a subset of artificial intelligence (AI) that enables 
        computers to learn and make decisions from data without being explicitly 
        programmed for every task.
        
        Types of Machine Learning:
        1. Supervised Learning - Uses labeled training data
        2. Unsupervised Learning - Finds patterns in unlabeled data
        3. Reinforcement Learning - Learns through interaction with environment
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            # Step 1: Process document
            document_id, chunks, metadata = await doc_processor.process_document(
                temp_path, "ml_guide.txt"
            )
            
            assert document_id is not None
            assert len(chunks) > 0
            assert metadata["filename"] == "ml_guide.txt"
            
            # Step 2: Store in vector database (mocked)
            success = await vector_store.add_documents(chunks)
            assert success is True
            
            # Step 3: Query the system
            response = await rag_service.generate_response(
                query="What is machine learning?",
                conversation_id="test_conv",
                use_context=True
            )
            
            assert response.response is not None
            assert len(response.response) > 0
            assert response.confidence_score > 0
            assert len(response.sources) > 0
            assert response.conversation_id == "test_conv"
            
        finally:
            # Cleanup
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_no_context_query(self, rag_pipeline):
        """Test querying without document context."""
        _, vector_store, rag_service = await rag_pipeline
        
        # Mock empty search results
        vector_store.similarity_search.return_value = []
        
        response = await rag_service.generate_response(
            query="What is quantum computing?",
            use_context=True
        )
        
        assert response.response is not None
        assert len(response.sources) == 0
        assert response.confidence_score >= 0
    
    @pytest.mark.asyncio
    async def test_conversation_continuity(self, rag_pipeline):
        """Test conversation history management."""
        _, _, rag_service = await rag_pipeline
        
        conv_id = "test_conversation"
        
        # First query
        response1 = await rag_service.generate_response(
            query="What is machine learning?",
            conversation_id=conv_id,
            use_context=True
        )
        
        # Second query (should have conversation context)
        response2 = await rag_service.generate_response(
            query="Can you explain it more simply?",
            conversation_id=conv_id,
            use_context=True
        )
        
        assert response1.conversation_id == conv_id
        assert response2.conversation_id == conv_id
        
        # Check that conversation history exists
        history = rag_service.get_conversation_history(conv_id)
        assert len(history) == 4  # 2 user messages + 2 assistant messages
    
    @pytest.mark.asyncio
    async def test_error_handling(self, rag_pipeline):
        """Test error handling in the pipeline."""
        _, vector_store, rag_service = await rag_pipeline
        
        # Mock an API error
        with patch.object(rag_service, '_generate_openai_response') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            # Should not raise, but return error response
            response = await rag_service.generate_response(
                query="Test query",
                use_context=True
            )
            
            assert "error" in response.response.lower()
            assert response.confidence_score == 0.0