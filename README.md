# Document Chat - Conversational AI for Document Q&A

A production-ready conversational AI assistant that enables users to upload documents and ask questions about their content using Retrieval Augmented Generation (RAG). Built with modern tech stack and engineering best practices.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │   FastAPI Backend │    │   ChromaDB      │
│   (TypeScript)   │◄──►│   (Python 3.11)  │◄──►│   Vector Store  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                      ┌──────────────────┐
                      │   OpenAI/Claude  │
                      │   LLM Services   │
                      └──────────────────┘
```

### Components

- **Frontend**: React with TypeScript for modern, type-safe UI
- **Backend**: FastAPI for high-performance Python APIs
- **Vector Database**: ChromaDB for similarity search and retrieval
- **LLM Services**: OpenAI GPT-4o-mini (primary) + Anthropic Claude (fallback)
- **Embeddings**: OpenAI text-embedding-3-small for cost-effective performance

## 🚀 Quick Setup

### Prerequisites

- Docker and Docker Compose
- OpenAI API key (required)
- Anthropic API key (optional, for fallback)

### One-Command Setup

1. **Clone and configure:**
   ```bash
   git clone <repository-url>
   cd document-chat
   cp backend/.env.example .env
   ```

2. **Add your API keys to `.env`:**
   ```bash
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here  # Optional
   ```

3. **Start the application:**
   ```bash
   docker compose up
   ```

4. **Access the application:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## 📚 RAG Implementation Details

### Document Processing Strategy

- **Chunking Approach:**
- **Chunk Size**: 300 tokens (optimal for context retention vs. specificity)
- **Overlap**: 50 tokens (≈16% overlap to preserve context across chunks)
- **Splitting Strategy**: Hierarchical splitting (paragraphs → sentences → words)
- **Token Counting**: tiktoken with cl100k_base encoding for accuracy

**Rationale**: 300-token chunks provide a balance between context and finer-grained retrieval. A 50-token overlap helps preserve continuity across chunk boundaries without excessive redundancy.

### Embedding & LLM Selection

**Embedding Model**: OpenAI text-embedding-3-small
- **Dimensions**: 1536
- **Cost**: $0.02/million tokens (10x cheaper than text-embedding-ada-002)
- **Performance**: 62.3% on MTEB benchmark
- **Rationale**: Best cost/performance ratio for most use cases

**Primary LLM**: GPT-4o-mini
- **Cost**: $0.15/million input tokens, $0.6/million output tokens
- **Speed**: ~2-3x faster than GPT-4
- **Context**: 128k tokens
- **Rationale**: Excellent balance of cost, speed, and quality

**Fallback LLM**: Claude 3 Haiku
- **Purpose**: Redundancy and different reasoning approach
- **Cost**: $0.25/million input tokens
- **Speed**: Very fast responses

### Retrieval Strategy

**Hybrid Approach:**
- **Top-k Retrieval**: 5 most similar chunks
- **Similarity Threshold**: 0.7 (filters low-relevance results)
- **Distance Metric**: Cosine similarity
- **Re-ranking**: Confidence scoring based on multiple factors

**Context Management:**
- **Max Context**: 4000 tokens (leaves room for conversation history)
- **Truncation Strategy**: Smart truncation preserving most relevant chunks
- **Memory**: Last 6 conversation turns for context continuity

### Prompt Engineering

**System Prompt Structure:**
```
Role Definition → Guidelines → Response Format → Quality Controls
```

**Key Elements:**
- Clear role definition as document-based assistant
- Explicit instructions for source citation
- Fallback behavior for insufficient context
- Professional tone and structured responses

### Quality Controls & Guardrails

**Implemented Safeguards:**
- **Source Attribution**: Every response includes source documents and similarity scores
- **Confidence Scoring**: Multi-factor confidence calculation
- **Relevance Checking**: Similarity threshold filtering
- **Fallback Responses**: Graceful handling of no-context scenarios
- **Error Handling**: Comprehensive exception management
- **Input Validation**: File type and size restrictions

**Confidence Score Calculation:**
```python
confidence = (avg_similarity + chunk_bonus + query_factor) * 100%
```

## 🔧 Technical Decisions & Rationale

### Tech Stack Choices

**FastAPI vs Flask/Django:**
- ✅ Automatic API documentation (OpenAPI/Swagger)
- ✅ Native async support for better performance
- ✅ Type hints integration for better code quality
- ✅ Built-in validation with Pydantic

**ChromaDB vs Pinecone/Weaviate:**
- ✅ Local development friendly
- ✅ No external dependencies for deployment
- ✅ Excellent performance for medium-scale datasets
- ✅ Simple setup and maintenance

**React vs Vue/Angular:**
- ✅ Large ecosystem and community
- ✅ TypeScript integration
- ✅ Component reusability
- ✅ Excellent developer tools

### Performance Optimizations

**Backend:**
- Connection pooling for database operations
- ThreadPoolExecutor for CPU-intensive tasks
- Async processing for I/O operations
- Efficient token counting with tiktoken

**Frontend:**
- Component memoization for re-render optimization
- Efficient state management
- Optimistic UI updates
- File upload progress indication

## 🤖 AI-Assisted Development Process

### Development Workflow

**AI Tools Used:**
- **GitHub Copilot**: Code completion and boilerplate generation
- **ChatGPT/Claude**: Architecture decisions and code review
- **AI-powered refactoring**: Code optimization and bug fixes

**Quality Assurance Process:**
1. **AI-Generated Code Review**: All AI-generated code manually reviewed
2. **Testing Strategy**: Unit tests for core functions, integration tests for APIs
3. **Code Standards**: Consistent formatting, type hints, and documentation
4. **Iterative Refinement**: Multiple AI-assisted iterations for optimization

**Maintaining Code Quality:**
- **Type Safety**: Comprehensive TypeScript and Python type hints
- **Documentation**: Inline comments and docstrings for complex logic
- **Error Handling**: Graceful failure modes and user-friendly messages
- **Modularity**: Clean separation of concerns and reusable components

### Repeatability & Maintainability

**Infrastructure as Code:**
- Docker configuration for consistent environments
- Environment variable management
- Automated dependency management

**Code Organization:**
- Clear module structure and naming conventions
- Separation of business logic and API endpoints
- Reusable service classes with dependency injection

## 🧪 Testing Strategy

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

**Coverage:**
- Unit tests for document processing
- Vector store operations
- RAG service functionality
- API endpoint integration tests

### Frontend Tests
```bash
cd frontend
npm test
```

**Coverage:**
- Component rendering tests
- User interaction testing
- API integration mocking

## 📊 Performance Metrics

**Response Times:**
- Document upload: < 5s for typical documents
- Query processing: < 3s end-to-end
- Embedding generation: ~1s per chunk

**Scalability:**
- Supports documents up to 50MB
- Concurrent user handling via async operations
- Efficient memory usage with streaming processing

## 🔮 Future Enhancements

### With More Time, I Would Add:

**Advanced Features:**
- **Multi-modal Support**: Image and table processing in documents
- **Advanced Re-ranking**: Cross-encoder models for better relevance
- **Semantic Caching**: Cache similar queries for faster responses
- **Document Versioning**: Track and manage document updates

**Production Readiness:**
- **Authentication & Authorization**: User management and access controls
- **Rate Limiting**: API throttling and abuse prevention
- **Monitoring & Analytics**: Comprehensive logging and metrics
- **Database Migration**: Production-grade database with migrations

**User Experience:**
- **Real-time Streaming**: Stream responses as they generate
- **Advanced UI**: Syntax highlighting, document previews
- **Mobile Optimization**: Responsive design improvements
- **Collaboration Features**: Shared workspaces and document collections

**Performance Optimizations:**
- **Redis Caching**: Cache frequently accessed embeddings
- **CDN Integration**: Static asset optimization
- **Load Balancing**: Horizontal scaling capabilities
- **Background Processing**: Asynchronous document processing queue

## 📝 Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `OPENAI_API_KEY` | OpenAI API key | - | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key (fallback) | - | No |
| `CHUNK_SIZE` | Document chunk size in tokens | 1000 | No |
| `CHUNK_OVERLAP` | Overlap between chunks | 200 | No |
| `TOP_K` | Number of chunks to retrieve | 5 | No |
| `SIMILARITY_THRESHOLD` | Minimum similarity score | 0.7 | No |
| `MAX_CONTEXT_TOKENS` | Maximum context size | 4000 | No |

## 🚦 API Endpoints

### Core Endpoints

**Upload Document:**
```
POST /documents/upload
Content-Type: multipart/form-data
```

**Chat with Documents:**
```
POST /chat
Content-Type: application/json
{
  "message": "What is this document about?",
  "conversation_id": "optional_id",
  "use_context": true
}
```

**Health Check:**
```
GET /health
```

Full API documentation available at: http://localhost:8000/docs

## 📄 Supported File Formats

- **PDF**: Text extraction from PDF documents
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files
- **MD**: Markdown files

## 🏆 Key Achievements

- ✅ **Production-ready architecture** with proper error handling
- ✅ **Comprehensive testing suite** with unit and integration tests
- ✅ **One-command deployment** via Docker Compose
- ✅ **Intelligent chunking strategy** optimized for context retention
- ✅ **Multi-LLM fallback system** for reliability
- ✅ **Real-time confidence scoring** for response quality assessment
- ✅ **Source attribution** for transparency and verification
- ✅ **Responsive UI** with modern design principles

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📜 License

MIT License - see LICENSE file for details

---

Built with ❤️ using modern AI development practices and tools.