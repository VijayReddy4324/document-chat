"""
Document processing service for parsing, chunking, and extracting metadata.
"""

import os
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path
import tiktoken
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Document parsing imports
try:
    import PyPDF2
    from docx import Document
except ImportError:
    PyPDF2 = None
    Document = None

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentProcessor:
    """Handles document parsing, chunking, and metadata extraction."""
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # Initialize text splitter with token-based chunking
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=encoding_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    async def parse_document(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """
        Parse document and extract text with metadata.
        Supports PDF, DOCX, and TXT files.
        """
        file_extension = Path(filename).suffix.lower()
        
        # Run parsing in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        if file_extension == '.pdf':
            text, metadata = await loop.run_in_executor(
                self.executor, self._parse_pdf, file_path, filename
            )
        elif file_extension == '.docx':
            text, metadata = await loop.run_in_executor(
                self.executor, self._parse_docx, file_path, filename
            )
        elif file_extension in ['.txt', '.md']:
            text, metadata = await loop.run_in_executor(
                self.executor, self._parse_text, file_path, filename
            )
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        return text, metadata
    
    def _parse_pdf(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Parse PDF file and extract text with metadata."""
        if PyPDF2 is None:
            raise ImportError("PyPDF2 is required for PDF processing")
        
        text = ""
        metadata = {
            "filename": filename,
            "file_type": "pdf",
            "total_pages": 0,
            "file_size": os.path.getsize(file_path)
        }
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    raise ValueError("PDF file is password-protected. Please provide an unencrypted PDF.")
                
                metadata["total_pages"] = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            text += f"\n\n--- Page {page_num} ---\n{page_text}"
                    except Exception as e:
                        print(f"Warning: Could not extract text from page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            if "decrypted" in str(e).lower():
                raise ValueError("PDF file is password-protected or corrupted. Please provide an unencrypted PDF.")
            else:
                raise ValueError(f"Error processing PDF file: {e}")
        
        if not text.strip():
            raise ValueError("No readable text found in PDF. The file may be image-based or corrupted.")
        
        return text.strip(), metadata
    
    def _parse_docx(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Parse DOCX file and extract text with metadata."""
        if Document is None:
            raise ImportError("python-docx is required for DOCX processing")
        
        doc = Document(file_path)
        
        paragraphs = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
        
        text = "\n\n".join(paragraphs)
        
        metadata = {
            "filename": filename,
            "file_type": "docx",
            "total_paragraphs": len(paragraphs),
            "file_size": os.path.getsize(file_path)
        }
        
        return text, metadata
    
    def _parse_text(self, file_path: str, filename: str) -> Tuple[str, Dict[str, Any]]:
        """Parse text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        metadata = {
            "filename": filename,
            "file_type": "text",
            "file_size": os.path.getsize(file_path)
        }
        
        return text, metadata
    
    async def create_chunks(
        self, 
        text: str, 
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split document text into chunks with metadata.
        Returns list of chunk dictionaries.
        """
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        chunk_documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": f"{metadata.get('filename', 'unknown')}_{i}",
                "chunk_index": i,
                "chunk_size": self.count_tokens(chunk),
                "total_chunks": len(chunks)
            })
            
            chunk_documents.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
        
        return chunk_documents
    
    async def process_document(
        self, 
        file_path: str, 
        filename: str
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Complete document processing pipeline.
        Returns: (document_id, chunks, metadata)
        """
        # Generate unique document ID
        document_id = str(uuid.uuid4())
        
        # Parse document
        text, metadata = await self.parse_document(file_path, filename)
        metadata["document_id"] = document_id
        metadata["token_count"] = self.count_tokens(text)
        
        # Create chunks
        chunks = await self.create_chunks(text, metadata)
        
        return document_id, chunks, metadata
    
    async def cleanup(self):
        """Cleanup resources."""
        self.executor.shutdown(wait=True)