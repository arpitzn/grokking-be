# Hackathon AI Agent Backend

A production-inspired, enterprise-grade AI agent backend featuring industry-standard LangGraph orchestration, real-time token-by-token streaming, sophisticated three-tier memory architecture, cloud-native RAG with Elasticsearch, and comprehensive observability via Langfuse.

## Features

- **LangGraph Orchestration**: Industry-standard agent framework with planner-executor pattern
- **Token-by-Token Streaming**: ChatGPT-level UX with real-time SSE responses
- **Three-Tier Memory**: Episodic (MongoDB), Working (runtime), Semantic (Mem0)
- **Intelligent Routing**: Planner classifies queries (internal RAG vs external search vs none)
- **Elasticsearch RAG**: Enterprise-grade vector search with multi-tenant design
- **NeMo Guardrails**: Production-grade safety (input/output/dialog rails)
- **Langfuse Observability**: Comprehensive tracing and monitoring
- **Parallel Execution**: Efficient use of `langgraph.types.Send` for reduced latency

## Quick Start

### Prerequisites

- Python 3.11+
- MongoDB Atlas account
- Elasticsearch Cloud account
- OpenAI API key
- Langfuse account
- Mem0 account (optional)

### Installation

1. Clone the repository:
```bash
git clone <repo-url>
cd hackathon-grokking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

### Docker

```bash
docker build -t hackathon-agent .
docker run -p 8000:8000 --env-file .env hackathon-agent
```

## API Endpoints

### Chat Streaming
- `POST /chat/stream` - Stream chat responses (SSE)

### Knowledge Management
- `POST /knowledge/upload` - Upload documents for RAG
- `GET /knowledge/{user_id}` - List user documents

### Conversations
- `GET /threads/{user_id}` - List user conversations
- `GET /threads/{conversation_id}/messages` - Get conversation messages

### Health
- `GET /health` - Health check for all dependencies

## Architecture

See `.cursor/plans/hackathon_agent_architecture_cf2870f5.plan.md` for detailed architecture documentation.

## Testing

```bash
pytest tests/
```

## License

See LICENSE file.
