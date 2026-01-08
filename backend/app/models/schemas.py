"""
Pydantic models for request/response schemas and data validation.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str
    filename: str
    status: str
    message: str
    chunks_created: int


class ChatMessage(BaseModel):
    """Individual chat message model."""
    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=2000)
    conversation_id: Optional[str] = None
    use_context: bool = Field(default=True, description="Whether to use document context")


class SourceReference(BaseModel):
    """Source reference for retrieved context."""
    document_name: str
    chunk_id: str
    page_number: Optional[int] = None
    similarity_score: float
    content_preview: str = Field(..., max_length=200)


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    conversation_id: str
    sources: List[SourceReference]
    confidence_score: float = Field(..., ge=0, le=1)
    reasoning: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DocumentSummary(BaseModel):
    """Summary of an uploaded document."""
    document_id: str
    filename: str
    upload_date: datetime
    size_bytes: int
    chunk_count: int
    status: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    dependencies: Dict[str, str]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)