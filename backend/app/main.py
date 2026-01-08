"""
FastAPI application with document upload and chat endpoints.
"""

import os
import sys
import logging
import uuid
import aiofiles
from pathlib import Path
from typing import List, Optional

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv

from app.models.schemas import (
    DocumentUploadResponse, ChatRequest, ChatResponse, 
    DocumentSummary, HealthResponse, ErrorResponse
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.rag_service import RAGService

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global service instances
document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global document_processor, vector_store, rag_service
    
    # Initialize services
    logger.info("Initializing services...")
    
    # Configuration from environment
    chunk_size = int(os.getenv("CHUNK_SIZE", 300))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 50))
    chroma_db_path = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
    top_k = int(os.getenv("TOP_K", 5))
    similarity_threshold = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))
    max_context_tokens = int(os.getenv("MAX_CONTEXT_TOKENS", 4000))
    
    # Initialize document processor
    document_processor = DocumentProcessor(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    # Initialize vector store
    vector_store = VectorStore(
        persist_directory=chroma_db_path
    )
    
    # Initialize RAG service using environment variables
    rag_service = RAGService(
        vector_store=vector_store,
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        fallback_model=os.getenv("FALLBACK_MODEL", "claude-3-haiku-20240307"),
        max_context_tokens=max_context_tokens,
        max_tokens=int(os.getenv("MAX_TOKENS", 1500)),
        temperature=float(os.getenv("TEMPERATURE", 0.7))
    )
    
    logger.info("Services initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Cleaning up services...")
    if document_processor:
        await document_processor.cleanup()
    if vector_store:
        await vector_store.cleanup()
    logger.info("Cleanup completed")


# Create FastAPI app
app = FastAPI(
    title="Document Chat API",
    description="A conversational AI assistant for document-based Q&A using RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency to get services
def get_document_processor() -> DocumentProcessor:
    if document_processor is None:
        raise HTTPException(status_code=500, detail="Document processor not initialized")
    return document_processor


def get_vector_store() -> VectorStore:
    if vector_store is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")
    return vector_store


def get_rag_service() -> RAGService:
    if rag_service is None:
        raise HTTPException(status_code=500, detail="RAG service not initialized")
    return rag_service


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0",
        dependencies={
            "openai": "1.3.0",
            "chromadb": "0.4.18",
            "fastapi": "0.104.1"
        }
    )


@app.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    vec_store: VectorStore = Depends(get_vector_store)
):
    """Upload and process a document."""
    try:
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.txt', '.md'}
        file_extension = Path(file.filename or "").suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create upload directory
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / f"{uuid.uuid4()}_{file.filename}"
        
        async with aiofiles.open(file_path, 'wb') as buffer:
            content = await file.read()
            await buffer.write(content)
        # Process document
        document_id, chunks, metadata = await doc_processor.process_document(
            str(file_path), file.filename or "unknown"
        )
        
        # Store in vector database
        success = await vec_store.add_documents(chunks)
        
        if not success:
            # Clean up file on failure
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail="Failed to store document in vector database")
        
        # Clean up temporary file
        file_path.unlink(missing_ok=True)
        
        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename or "unknown",
            status="success",
            message="Document uploaded and processed successfully",
            chunks_created=len(chunks)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Chat with documents using RAG."""
    try:
        response = await rag_svc.generate_response(
            query=request.message,
            conversation_id=request.conversation_id,
            use_context=request.use_context
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents", response_model=List[DocumentSummary])
async def list_documents(
    vec_store: VectorStore = Depends(get_vector_store)
):
    """List all uploaded documents."""
    try:
        # Get collection stats
        stats = vec_store.get_collection_stats()
        
        # For now, return basic stats (in production, implement proper document tracking)
        return [DocumentSummary(
            document_id="all",
            filename="Combined Documents",
            upload_date=datetime.utcnow(),
            size_bytes=0,
            chunk_count=stats.get("total_chunks", 0),
            status="active"
        )]
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations/{conversation_id}")
async def clear_conversation(
    conversation_id: str,
    rag_svc: RAGService = Depends(get_rag_service)
):
    """Clear conversation history."""
    try:
        success = rag_svc.clear_conversation(conversation_id)
        
        if success:
            return {"message": "Conversation cleared successfully"}
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail="An unexpected error occurred"
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )