"""FastAPI application entry point"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, knowledge, health, threads
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hackathon AI Agent Backend",
    description="Production-inspired AI agent with LangGraph, streaming, and observability",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For POC, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat.router)
app.include_router(knowledge.router)
app.include_router(health.router)
app.include_router(threads.router)


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Hackathon AI Agent Backend...")
    
    # Initialize MongoDB connection
    from app.infra.mongo import get_mongodb_client
    await get_mongodb_client()
    logger.info("MongoDB client initialized")
    
    # Initialize Elasticsearch
    from app.infra.elasticsearch import get_elasticsearch_client
    await get_elasticsearch_client()
    logger.info("Elasticsearch client initialized")
    
    # Initialize LangGraph
    from app.agent.graph import get_graph
    get_graph()
    logger.info("LangGraph initialized")
    
    # Initialize Langfuse CallbackHandler
    from app.infra.langfuse_callback import langfuse_handler
    logger.info("Langfuse CallbackHandler initialized")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down...")
    
    # Close MongoDB connection
    try:
        from app.infra.mongo import mongodb_client
        if mongodb_client:
            await mongodb_client.close()
    except Exception as e:
        logger.warning(f"Error closing MongoDB: {e}")
    
    # Close Elasticsearch connection
    try:
        from app.infra.elasticsearch import _elasticsearch_client
        if _elasticsearch_client:
            await _elasticsearch_client.close()
    except Exception as e:
        logger.warning(f"Error closing Elasticsearch: {e}")
    
    # Close Mem0 connection
    try:
        from app.infra.mem0 import mem0_client
        if mem0_client:
            await mem0_client.close()
    except Exception as e:
        logger.warning(f"Error closing Mem0: {e}")
    
    logger.info("Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hackathon AI Agent Backend",
        "version": "0.1.0",
        "status": "running"
    }
