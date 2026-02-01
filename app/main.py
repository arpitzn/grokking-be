"""FastAPI application entry point"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import chat, knowledge, health, threads, escalations, memory, users, escalated_tickets, zones, restaurants, orders
import logging
import json
import asyncio

# Configure JSON logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        # If message is already JSON, pass through
        try:
            json.loads(record.getMessage())
            return record.getMessage()
        except (json.JSONDecodeError, ValueError):
            # Otherwise, wrap in JSON
            log_data = {
                "timestamp": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage()
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler()]
)
for handler in logging.root.handlers:
    handler.setFormatter(JSONFormatter())

# Reduce NeMo Guardrails logging verbosity
# Set to WARNING to suppress INFO/DEBUG messages (config dumps, runtime events, etc.)
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails.rails").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails.colang").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails.actions").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails.library").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    logger.info("Starting Hackathon AI Agent Backend...")
    
    # Import all services
    from app.infra.mongo import get_mongodb_client
    from app.infra.elasticsearch import get_elasticsearch_client
    from app.agent.graph import get_graph
    from app.infra.langfuse_callback import langfuse_handler
    from app.infra.guardrails import get_guardrails_manager
    from app.infra.mem0 import get_mem0_client
    
    # Initialize MongoDB connection
    await get_mongodb_client()
    logger.info("MongoDB client initialized")
    
    # Initialize Elasticsearch
    await get_elasticsearch_client()
    logger.info("Elasticsearch client initialized")
    
    # Initialize LangGraph
    get_graph()
    logger.info("LangGraph initialized")
    
    # Initialize Langfuse CallbackHandler
    logger.info("Langfuse CallbackHandler initialized")
    
    # Initialize Guardrails Manager
    get_guardrails_manager()
    logger.info("Guardrails Manager initialized")
    
    # Validate Mem0 connection
    try:
        mem0_client = await get_mem0_client()
        logger.info("Mem0 client initialized")
    except Exception as e:
        logger.warning(f"Mem0 initialization failed: {e} - service will be degraded")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down...")
    
    # Cancel all pending tasks before closing connections
    try:
        # Get current event loop
        loop = asyncio.get_running_loop()
        # Get all pending tasks (excluding current task)
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done() and task is not asyncio.current_task()]
        if pending_tasks:
            logger.info(f"Cancelling {len(pending_tasks)} pending tasks...")
            # Cancel all pending tasks
            for task in pending_tasks:
                task.cancel()
            
            # Wait for tasks to complete cancellation (with timeout)
            await asyncio.gather(*pending_tasks, return_exceptions=True)
            logger.info("All pending tasks cancelled")
    except Exception as e:
        logger.warning(f"Error cancelling pending tasks: {e}")
    
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
        from app.infra.mem0 import mem0_service
        if mem0_service:
            await mem0_service.close()
    except Exception as e:
        logger.warning(f"Error closing Mem0: {e}")
    
    logger.info("Shutdown complete")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Hackathon AI Agent Backend",
    description="Production-inspired AI agent with LangGraph, streaming, and observability",
    version="0.1.0",
    lifespan=lifespan
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
app.include_router(escalations.router)
app.include_router(memory.router)
app.include_router(users.router)
app.include_router(escalated_tickets.router)
app.include_router(zones.router)
app.include_router(restaurants.router)
app.include_router(orders.router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Hackathon AI Agent Backend",
        "version": "0.1.0",
        "status": "running"
    }
