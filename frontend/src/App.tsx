import React, { useState, useRef, useEffect } from 'react';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  sources?: SourceReference[];
  confidence_score?: number;
}

interface SourceReference {
  document_name: string;
  chunk_id: string;
  page_number?: number;
  similarity_score: number;
  content_preview: string;
}

interface MessageMetaProps {
  timestamp: string;
  confidence_score?: number;
}

interface SourceItemProps {
  source: SourceReference;
  sourceIndex: number;
}

interface SourcesListProps {
  sources: SourceReference[];
}

interface MessageItemProps {
  message: ChatMessage;
  index: number;
  formatTimestamp: (timestamp: string) => string;
}

// Component definitions
const MessageMeta: React.FC<MessageMetaProps> = ({ timestamp, confidence_score }) => (
  <div className="message-meta">
    {new Date(timestamp).toLocaleTimeString()}
    {confidence_score && (
      <span className="confidence">
        • Confidence: {(confidence_score * 100).toFixed(0)}%
      </span>
    )}
  </div>
);

const SourceItem: React.FC<SourceItemProps> = ({ source, sourceIndex }) => (
  <div key={sourceIndex} className="source">
    📄 {source.document_name} 
    <span className="similarity">
      ({(source.similarity_score * 100).toFixed(0)}% match)
    </span>
    <div className="source-preview">{source.content_preview}</div>
  </div>
);

const SourcesList: React.FC<SourcesListProps> = ({ sources }) => (
  <div className="sources">
    <strong>Sources:</strong>
    {sources.map((source: SourceReference, sourceIndex: number) => (
      <SourceItem key={sourceIndex} source={source} sourceIndex={sourceIndex} />
    ))}
  </div>
);

const MessageItem: React.FC<MessageItemProps> = ({ message, index, formatTimestamp }) => (
  <div key={index} className={`message ${message.role}`}>
    <div className="message-content">
      <div className="message-text">{message.content}</div>
      <MessageMeta timestamp={message.timestamp} confidence_score={message.confidence_score} />
      {message.sources && message.sources.length > 0 && (
        <SourcesList sources={message.sources} />
      )}
    </div>
  </div>
);

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello! I\'m your document assistant. Upload some documents and I\'ll help you answer questions about them.',
      timestamp: new Date().toISOString()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId] = useState(() => `conv_${Date.now()}`);
  const [uploadingFile, setUploadingFile] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const uploadDocument = async (file: File) => {
    setUploadingFile(true);
    
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `✅ Successfully uploaded and processed "${file.name}". Created ${result.chunks_created} chunks. You can now ask questions about this document!`,
        timestamp: new Date().toISOString()
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `❌ Error uploading file: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setUploadingFile(false);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      uploadDocument(file);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: inputMessage,
          conversation_id: conversationId,
          use_context: true
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const result = await response.json();
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: result.response,
        timestamp: result.timestamp,
        sources: result.sources,
        confidence_score: result.confidence_score
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>📚 Document Chat</h1>
        <p>Upload documents and chat with them using AI</p>
      </header>

      <div className="chat-container">
        <div className="upload-section">
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".pdf,.docx,.txt,.md"
            style={{ display: 'none' }}
          />
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadingFile}
          >
            {uploadingFile ? '⏳ Uploading...' : '📤 Upload Document'}
          </button>
        </div>

        <div className="messages-container">
          {messages.map((message: ChatMessage, index: number) => (
            <MessageItem 
              key={index} 
              message={message} 
              index={index} 
              formatTimestamp={formatTimestamp} 
            />
          ))}
          {isLoading && (
            <div className="message assistant loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-section">
          <div className="input-container">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="message-input"
              rows={3}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className="send-btn"
              disabled={!inputMessage.trim() || isLoading}
            >
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;