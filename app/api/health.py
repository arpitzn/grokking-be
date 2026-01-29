"""
Health check endpoint - Comprehensive dependency monitoring

This endpoint validates all critical services our AI agent depends on:
- MongoDB: Primary database for conversations and messages
- Elasticsearch: Vector search for RAG (Retrieval Augmented Generation)
- Mem0: Semantic memory for long-term context
- Langfuse: LLM observability and tracing
- OpenAI: Core LLM provider for chat completions

Why this matters: Health checks help us catch issues early and ensure
our AI agent is ready to serve users reliably! 🚀
"""
from fastapi import APIRouter
from app.models.schemas import HealthCheckResponse
from app.infra.mongo import get_mongodb_client
from app.infra.elasticsearch import get_elasticsearch_client
from app.infra.mem0 import get_mem0_client
from app.infra.langfuse import get_langfuse_client
from app.infra.llm import get_llm_client
from datetime import datetime
import logging
import asyncio
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Timeout for health checks - prevents hanging if services are slow
# 5 seconds is a good balance: fast enough for users, generous enough for network delays
HEALTH_CHECK_TIMEOUT = 5.0


async def with_timeout(coro, timeout: float, service_name: str):
    """
    Wrapper to add timeout protection to async health checks
    
    Why this is important: Without timeouts, a single slow service could
    make our entire health endpoint hang, which is bad for monitoring!
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(f"{service_name} health check timed out after {timeout}s")
        raise TimeoutError(f"Health check timed out")


async def check_service(coro, timeout: float, service_name: str, default_status: str = "unhealthy"):
    """
    Helper function for consistent error handling across all health checks
    
    This DRY (Don't Repeat Yourself) approach makes our code cleaner and
    ensures all services get the same quality of error handling. Plus,
    it's easier to maintain if we need to change error handling logic!
    """
    try:
        result = await with_timeout(coro, timeout, service_name)
        return {"status": "healthy", "message": result if isinstance(result, str) else "Connected"}
    except TimeoutError:
        return {"status": default_status, "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"{service_name} health check failed: {e}")
        return {"status": default_status, "message": "Connection failed"}


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check for all dependencies
    
    This is our "canary in the coal mine" - it tells us if everything
    is working before users hit issues. Perfect for monitoring dashboards!
    
    Status levels:
        - healthy: All systems go! 🟢
        - degraded: Some non-critical services down, but core works 🟡
        - unhealthy: Critical services down, agent may not work 🔴
    
    Returns detailed status for each service so we can debug quickly.
    """
    checks = {}
    
    # MongoDB check - Our primary database for storing conversations
    # Using ping() is lightweight and validates connectivity without heavy queries
    async def check_mongodb():
        client = await get_mongodb_client()
        healthy = await client.ping()
        return "Connected" if healthy else "Connection failed"
    
    result = await check_service(check_mongodb(), 2.0, "MongoDB")
    checks["mongodb"] = result
    
    # Elasticsearch check - Our vector database for RAG (knowledge retrieval)
    # Returns rich cluster info because ES health is more complex than simple ping
    # This helps us debug if vector search isn't working properly
    async def check_elasticsearch():
        client = await get_elasticsearch_client()
        return await client.health_check()
    
    try:
        health_info = await with_timeout(check_elasticsearch(), 3.0, "Elasticsearch")
        # Elasticsearch gives us lots of useful info - let's pass it through!
        checks["elasticsearch"] = {
            "status": health_info["status"],
            "cluster_name": health_info.get("cluster_name"),
            "version": health_info.get("version"),
            "cluster_health": health_info.get("cluster_health"),  # green/yellow/red
            "nodes": health_info.get("number_of_nodes"),
            "auth_configured": health_info.get("authentication_configured"),
            "message": health_info.get("error", "Connected")
        }
    except TimeoutError:
        checks["elasticsearch"] = {"status": "unhealthy", "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"Elasticsearch health check failed: {e}")
        checks["elasticsearch"] = {"status": "unhealthy", "message": "Connection failed"}
    
    # Mem0 check - Semantic memory service for long-term context
    # Marked as "degraded" (not unhealthy) because agent can work without it
    # This way we know there's an issue but don't panic - graceful degradation! 🎯
    async def check_mem0():
        client = await get_mem0_client()
        return await client.health_check()
    
    try:
        mem0_healthy = await with_timeout(check_mem0(), 3.0, "Mem0")
        checks["mem0"] = {
            "status": "healthy" if mem0_healthy else "degraded",
            "message": "API accessible" if mem0_healthy else "API unreachable"
        }
    except TimeoutError:
        checks["mem0"] = {"status": "degraded", "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"Mem0 health check failed: {e}")
        checks["mem0"] = {"status": "degraded", "message": "API unreachable"}
    
    # Langfuse check - LLM observability platform for tracing and monitoring
    # We validate credentials because wrong keys = no observability = harder debugging!
    # Using auth_check() if available, otherwise fallback to config check
    async def check_langfuse():
        langfuse = get_langfuse_client()
        if hasattr(langfuse, 'auth_check'):
            # Run in thread pool since auth_check might be synchronous
            return await asyncio.to_thread(langfuse.auth_check)
        # Fallback: at least verify credentials are configured
        if hasattr(langfuse, 'public_key') and hasattr(langfuse, 'secret_key'):
            return bool(langfuse.public_key and langfuse.secret_key)
        return True
    
    try:
        auth_valid = await with_timeout(check_langfuse(), 3.0, "Langfuse")
        checks["langfuse"] = {
            "status": "healthy" if auth_valid else "degraded",
            "message": "Credentials valid" if auth_valid else "Authentication failed"
        }
    except TimeoutError:
        checks["langfuse"] = {"status": "degraded", "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"Langfuse health check failed: {e}")
        # Smart error detection: check if it's an auth issue vs network issue
        error_msg = str(e).lower()
        is_auth_error = any(x in error_msg for x in ["401", "403", "unauthorized", "authentication", "invalid"])
        checks["langfuse"] = {
            "status": "degraded",
            "message": "Invalid credentials" if is_auth_error else "Service unreachable"
        }
    
    # OpenAI check - The heart of our AI agent! ❤️
    # We make a real API call (not just check config) because wrong keys fail silently
    # Using max_tokens=1 to minimize cost - this is just a health check after all!
    async def check_openai():
        llm_client = get_llm_client()
        # Minimal API call: just enough to validate credentials work
        await llm_client.client.ainvoke(
            [HumanMessage(content="test")],
            config={"max_tokens": 1}  # Super cheap - just 1 token!
        )
        return "API key valid"
    
    try:
        result = await with_timeout(check_openai(), 5.0, "OpenAI")
        checks["openai"] = {"status": "healthy", "message": result}
    except TimeoutError:
        checks["openai"] = {"status": "unhealthy", "message": "Connection timeout"}
    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        # Detect auth errors vs network issues for better debugging
        error_msg = str(e).lower()
        is_auth_error = any(x in error_msg for x in ["invalid", "authentication", "api key"])
        checks["openai"] = {
            "status": "unhealthy",
            "message": "API key invalid" if is_auth_error else "Service unreachable"
        }
    
    # Determine overall status - prioritize unhealthy > degraded > healthy
    # This way, if ANY critical service is down, we know immediately!
    status = "healthy"
    if any(c["status"] == "unhealthy" for c in checks.values()):
        status = "unhealthy"  # Critical services down - agent won't work
    elif any(c["status"] == "degraded" for c in checks.values()):
        status = "degraded"  # Some services down, but core functionality works
    
    return HealthCheckResponse(
        status=status,
        checks=checks,
        timestamp=datetime.utcnow().isoformat()  # When this check ran
    )
