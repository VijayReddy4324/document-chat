"""
RAG (Retrieval Augmented Generation) service for conversational AI.
Handles query processing, context retrieval, and response generation.
"""

import os
import logging
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import anthropic
from datetime import datetime
import json

from app.services.vector_store import VectorStore
from app.models.schemas import ChatResponse, SourceReference

logger = logging.getLogger(__name__)


class RAGService:
    """RAG service for processing queries and generating responses."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        openai_api_key: str = None,
        anthropic_api_key: Optional[str] = None,
        default_model: str = "gpt-4o-mini",
        fallback_model: str = "claude-3-haiku-20240307",
        max_context_tokens: int = 4000,
        max_tokens: int = 1500,
        temperature: float = 0.7
    ):
        self.vector_store = vector_store
        self.default_model = default_model
        self.fallback_model = fallback_model
        self.max_context_tokens = max_context_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Initialize LLM clients using environment variables
        self.openai_client = OpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
        )
        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.anthropic_client = None
        
        # Initialize tokenizer
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Conversation memory (in production, use Redis/database)
        self.conversations: Dict[str, List[Dict[str, str]]] = {}
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the assistant."""
        return """You are a helpful AI assistant that answers questions based on provided document context. 

Key guidelines:
1. Use only the provided context to answer questions
2. If the context doesn't contain enough information, clearly state this limitation
3. Always cite your sources using the document names provided
4. Be precise and factual, avoiding speculation
5. If asked about topics not covered in the context, politely redirect to document-related questions
6. Provide clear, well-structured responses
7. When appropriate, suggest follow-up questions related to the documents

Your responses should be:
- Accurate and grounded in the provided context
- Well-structured and easy to read
- Appropriately detailed for the question asked
- Professional and helpful in tone"""
    
    def _format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, List[SourceReference]]:
        """Format retrieved chunks into context string and extract sources."""
        if not retrieved_chunks:
            return "", []
        
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            content = chunk["content"]
            metadata = chunk["metadata"]
            similarity_score = chunk["similarity_score"]
            
            # Create context entry
            source_name = metadata.get("filename", "Unknown Document")
            context_parts.append(f"[Source {i}: {source_name}]\n{content}\n")
            
            # Create source reference with properly truncated content_preview
            content_preview = content[:197] + "..." if len(content) > 200 else content
            source_ref = SourceReference(
                document_name=source_name,
                chunk_id=chunk["chunk_id"],
                page_number=metadata.get("page_number"),
                similarity_score=similarity_score,
                content_preview=content_preview
            )
            sources.append(source_ref)
        
        context = "\n".join(context_parts)
        return context, sources
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limit."""
        context_tokens = self.count_tokens(context)
        
        if context_tokens <= max_tokens:
            return context
        
        # Simple truncation - in production, use smarter strategies
        # like keeping the most relevant chunks
        words = context.split()
        truncated = ""
        
        for word in words:
            test_context = truncated + " " + word if truncated else word
            if self.count_tokens(test_context) > max_tokens:
                break
            truncated = test_context
        
        return truncated + "\n\n[Note: Context truncated due to length limits]"
    
    def _calculate_confidence_score(
        self, 
        retrieved_chunks: List[Dict[str, Any]], 
        query: str
    ) -> float:
        """Calculate confidence score based on retrieval quality."""
        if not retrieved_chunks:
            return 0.0
        
        # Simple confidence calculation based on similarity scores
        avg_similarity = sum(chunk["similarity_score"] for chunk in retrieved_chunks) / len(retrieved_chunks)
        
        # Boost confidence if we have multiple relevant chunks
        chunk_bonus = min(len(retrieved_chunks) * 0.1, 0.3)
        
        # Query length factor (longer queries often get better matches)
        query_factor = min(len(query.split()) / 10, 0.2)
        
        confidence = min(avg_similarity + chunk_bonus + query_factor, 1.0)
        return round(confidence, 2)
    
    async def _generate_openai_response(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, Optional[str]]:
        """Generate response using OpenAI."""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.choices[0].message.content
            reasoning = f"Generated using {self.default_model}"
            
            return content, reasoning
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e
    
    async def _generate_anthropic_response(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, Optional[str]]:
        """Generate response using Anthropic Claude as fallback."""
        if not self.anthropic_client:
            raise Exception("Anthropic client not configured")
        
        try:
            # Convert messages format for Anthropic
            system_prompt = None
            conversation = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_prompt = msg["content"]
                else:
                    conversation.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = self.anthropic_client.messages.create(
                model=self.fallback_model,
                system=system_prompt or "",
                messages=conversation,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            content = response.content[0].text
            reasoning = f"Generated using {self.fallback_model} (fallback)"
            
            return content, reasoning
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise e
    
    async def generate_response(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        use_context: bool = True,
        top_k: int = 5,
        similarity_threshold: float = 0.3
    ) -> ChatResponse:
        """
        Generate response for user query using RAG.
        """
        start_time = datetime.utcnow()
        
        try:
            # Retrieve relevant context if requested
            retrieved_chunks = []
            sources = []
            
            if use_context:
                # Check if we have any documents in the vector store
                stats = self.vector_store.get_collection_stats()
                logger.info(f"Vector store stats: {stats}")
                
                if stats.get('total_chunks', 0) == 0:
                    logger.warning("No documents found in vector store")
                else:
                    logger.info(f"Performing similarity search for query: '{query[:50]}...'")
                    retrieved_chunks = await self.vector_store.similarity_search(
                        query=query,
                        top_k=top_k,
                        similarity_threshold=similarity_threshold
                    )
                    logger.info(f"Retrieved {len(retrieved_chunks)} chunks with threshold {similarity_threshold}")
                    
                    # If no results, try with lower threshold
                    if not retrieved_chunks and similarity_threshold > 0.1:
                        logger.warning(f"No chunks found with threshold {similarity_threshold}, trying with 0.1")
                        retrieved_chunks = await self.vector_store.similarity_search(
                            query=query,
                            top_k=top_k,
                            similarity_threshold=0.1
                        )
                        logger.info(f"Retrieved {len(retrieved_chunks)} chunks with lower threshold 0.1")
            
            # Format context and sources
            context, sources = self._format_context(retrieved_chunks)
            
            # Prepare conversation history
            conversation_history = []
            if conversation_id and conversation_id in self.conversations:
                conversation_history = self.conversations[conversation_id][-6:]  # Keep last 6 messages
            
            # Build messages for LLM
            messages = [{"role": "system", "content": self._create_system_prompt()}]
            
            # Add conversation history
            for msg in conversation_history:
                messages.append(msg)
            
            # Add current query with context
            if context:
                truncated_context = self._truncate_context(context, self.max_context_tokens)
                user_message = f"Context:\n{truncated_context}\n\nQuestion: {query}"
            else:
                user_message = f"Question: {query}\n\nNote: No relevant document context found. Please answer based on general knowledge or ask for document upload."
            
            messages.append({"role": "user", "content": user_message})
            
            # Generate response with fallback
            try:
                response_content, reasoning = await self._generate_openai_response(messages)
            except Exception as e:
                logger.warning(f"OpenAI failed, trying Anthropic fallback: {e}")
                if self.anthropic_client:
                    response_content, reasoning = await self._generate_anthropic_response(messages)
                else:
                    raise e
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(retrieved_chunks, query)
            
            # Update conversation memory
            if conversation_id:
                if conversation_id not in self.conversations:
                    self.conversations[conversation_id] = []
                
                self.conversations[conversation_id].extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": response_content}
                ])
                
                # Keep conversation memory manageable
                if len(self.conversations[conversation_id]) > 20:
                    self.conversations[conversation_id] = self.conversations[conversation_id][-20:]
            
            # Create response
            response = ChatResponse(
                response=response_content,
                conversation_id=conversation_id or f"conv_{int(datetime.utcnow().timestamp())}",
                sources=sources,
                confidence_score=confidence_score,
                reasoning=reasoning,
                timestamp=start_time
            )
            
            logger.info(f"Generated response in {(datetime.utcnow() - start_time).total_seconds():.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            # Return error response
            return ChatResponse(
                response="I apologize, but I encountered an error while processing your question. Please try again or contact support if the issue persists.",
                conversation_id=conversation_id or f"error_{int(datetime.utcnow().timestamp())}",
                sources=[],
                confidence_score=0.0,
                reasoning=f"Error: {str(e)}",
                timestamp=start_time
            )
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            return True
        return False
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversations.get(conversation_id, [])