"""
Test configuration and fixtures.
"""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing."""
    mock_client = MagicMock()
    
    # Mock embeddings response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        MagicMock(embedding=[0.1] * 1536)  # Mock embedding vector
    ]
    mock_client.embeddings.create.return_value = mock_embedding_response
    
    # Mock chat completion response
    mock_completion_response = MagicMock()
    mock_completion_response.choices = [
        MagicMock(message=MagicMock(content="This is a test response"))
    ]
    mock_client.chat.completions.create.return_value = mock_completion_response
    
    return mock_client


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = AsyncMock()
    
    # Mock search results
    mock_store.similarity_search.return_value = [
        {
            "content": "Test document content",
            "metadata": {"filename": "test.txt", "chunk_id": "test_0"},
            "similarity_score": 0.85,
            "chunk_id": "test_0"
        }
    ]
    
    # Mock add documents
    mock_store.add_documents.return_value = True
    
    return mock_store


# Environment variables for testing
os.environ["OPENAI_API_KEY"] = "test_key"
os.environ["ANTHROPIC_API_KEY"] = "test_key"