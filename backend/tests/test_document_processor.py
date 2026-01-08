"""
Unit tests for the DocumentProcessor service.
"""

import pytest
import tempfile
import os
from pathlib import Path

from app.services.document_processor import DocumentProcessor


@pytest.fixture
def document_processor():
    """Create a DocumentProcessor instance for testing."""
    return DocumentProcessor(chunk_size=500, chunk_overlap=100)


@pytest.fixture
def sample_text_file():
    """Create a temporary text file for testing."""
    content = """This is a sample document for testing.
    
It contains multiple paragraphs with different content.
This should be sufficient to test the document processing functionality.

The document processor should be able to parse this text and create chunks
from it based on the specified chunk size and overlap parameters.
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(content)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


class TestDocumentProcessor:
    """Test cases for DocumentProcessor."""
    
    def test_initialization(self):
        """Test DocumentProcessor initialization."""
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        assert processor.chunk_size == 1000
        assert processor.chunk_overlap == 200
        assert processor.encoding is not None
    
    def test_count_tokens(self, document_processor):
        """Test token counting functionality."""
        text = "This is a test sentence."
        token_count = document_processor.count_tokens(text)
        assert isinstance(token_count, int)
        assert token_count > 0
    
    @pytest.mark.asyncio
    async def test_parse_text_document(self, document_processor, sample_text_file):
        """Test parsing a text document."""
        filename = Path(sample_text_file).name
        text, metadata = await document_processor.parse_document(sample_text_file, filename)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert isinstance(metadata, dict)
        assert metadata["filename"] == filename
        assert metadata["file_type"] == "text"
        assert "file_size" in metadata
    
    @pytest.mark.asyncio
    async def test_create_chunks(self, document_processor):
        """Test chunk creation from text."""
        text = "This is a long text that should be split into multiple chunks. " * 50
        metadata = {"filename": "test.txt", "file_type": "text"}
        
        chunks = await document_processor.create_chunks(text, metadata)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert "content" in chunk
            assert "metadata" in chunk
            assert "chunk_id" in chunk["metadata"]
            assert "chunk_index" in chunk["metadata"]
    
    @pytest.mark.asyncio
    async def test_process_document(self, document_processor, sample_text_file):
        """Test the complete document processing pipeline."""
        filename = Path(sample_text_file).name
        document_id, chunks, metadata = await document_processor.process_document(
            sample_text_file, filename
        )
        
        assert isinstance(document_id, str)
        assert len(document_id) > 0
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        assert isinstance(metadata, dict)
        assert "document_id" in metadata
        assert "token_count" in metadata
    
    @pytest.mark.asyncio
    async def test_unsupported_file_format(self, document_processor):
        """Test handling of unsupported file formats."""
        import tempfile
        import os
        
        # Create a temporary file with unsupported extension
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                await document_processor.parse_document(temp_path, "test.xyz")
        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)