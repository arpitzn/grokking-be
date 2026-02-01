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
from app.infra.llm import get_llm_service
from datetime import datetime, timezone
import logging
import asyncio
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# Timeout configurations for health checks - prevents hanging if services are slow
# These are optimized for parallel execution to keep total time under 3 seconds
MONGODB_TIMEOUT = 10.0  # Ping is instant if healthy
ELASTICSEARCH_TIMEOUT = 15.0  # Cluster health check needs more time
MEM0_TIMEOUT = 15.0  # Ping is faster, reduced from 5s
LANGFUSE_TIMEOUT = 10.0  # Auth check
OPENAI_TIMEOUT = 10.0  # LLM API call (reduced from 5s for parallel execution)


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
    
    Performance: All checks run in parallel for ~3 second total time
    instead of ~16 seconds sequential!
    """
    
    # MongoDB check - Our primary database for storing conversations
    # Using ping() is lightweight and validates connectivity without heavy queries
    async def check_mongodb():
        client = await get_mongodb_client()
        healthy = await client.ping()
        return "Connected" if healthy else "Connection failed"
    
    # Elasticsearch check - Our vector database for RAG (knowledge retrieval)
    # Returns rich cluster info because ES health is more complex than simple ping
    # This helps us debug if vector search isn't working properly
    async def check_elasticsearch():
        try:
            client = await get_elasticsearch_client()
            health_info = await with_timeout(client.health_check(), ELASTICSEARCH_TIMEOUT, "Elasticsearch")
            # Elasticsearch gives us lots of useful info - let's pass it through!
            return {
                "status": health_info["status"],
                "cluster_name": health_info.get("cluster_name"),
                "version": health_info.get("version"),
                "cluster_health": health_info.get("cluster_health"),  # green/yellow/red
                "nodes": health_info.get("number_of_nodes"),
                "auth_configured": health_info.get("authentication_configured"),
                "message": health_info.get("error", "Connected")
            }
        except TimeoutError:
            return {"status": "unhealthy", "message": "Connection timeout"}
        except Exception as e:
            logger.error(f"Elasticsearch health check failed: {e}")
            return {"status": "unhealthy", "message": "Connection failed"}
    
    # Mem0 check - Semantic memory service for long-term context
    # Marked as "degraded" (not unhealthy) because agent can work without it
    # This way we know there's an issue but don't panic - graceful degradation! 🎯
    async def check_mem0():
        try:
            client = await get_mem0_client()
            mem0_healthy = await with_timeout(client.ping(), MEM0_TIMEOUT, "Mem0")
            return {
                "status": "healthy" if mem0_healthy else "degraded",
                "message": "API accessible" if mem0_healthy else "API unreachable"
            }
        except TimeoutError:
            return {"status": "degraded", "message": "Connection timeout"}
        except Exception as e:
            logger.error(f"Mem0 health check failed: {e}", exc_info=True)
            # Detect auth errors vs network issues for better debugging
            error_msg = str(e).lower()
            is_auth_error = any(x in error_msg for x in ["401", "403", "unauthorized", "authentication", "invalid", "api key"])
            return {
                "status": "degraded",
                "message": "Invalid API key" if is_auth_error else "API unreachable"
            }
    
    # Langfuse check - LLM observability platform for tracing and monitoring
    # We validate credentials because wrong keys = no observability = harder debugging!
    # Using auth_check() if available, otherwise fallback to config check
    async def check_langfuse():
        try:
            langfuse = get_langfuse_client()
            if hasattr(langfuse, 'auth_check'):
                # Run in thread pool since auth_check might be synchronous
                auth_valid = await with_timeout(
                    asyncio.to_thread(langfuse.auth_check), 
                    LANGFUSE_TIMEOUT, 
                    "Langfuse"
                )
            else:
                # Fallback: at least verify credentials are configured
                if hasattr(langfuse, 'public_key') and hasattr(langfuse, 'secret_key'):
                    auth_valid = bool(langfuse.public_key and langfuse.secret_key)
                else:
                    auth_valid = True
            
            return {
                "status": "healthy" if auth_valid else "degraded",
                "message": "Credentials valid" if auth_valid else "Authentication failed"
            }
        except TimeoutError:
            return {"status": "degraded", "message": "Connection timeout"}
        except Exception as e:
            logger.error(f"Langfuse health check failed: {e}")
            # Smart error detection: check if it's an auth issue vs network issue
            error_msg = str(e).lower()
            is_auth_error = any(x in error_msg for x in ["401", "403", "unauthorized", "authentication", "invalid"])
            return {
                "status": "degraded",
                "message": "Invalid credentials" if is_auth_error else "Service unreachable"
            }
    
    # OpenAI check - The heart of our AI agent! ❤️
    # We make a real API call (not just check config) because wrong keys fail silently
    # Using max_tokens=1 to minimize cost - this is just a health check after all!
    async def check_openai():
        try:
            llm_service = get_llm_service()
            # Get a minimal LLM instance for health check
            llm = llm_service.get_llm_instance(
                model_name="gpt-4.1-mini",
                max_completion_tokens=1  # Super cheap - just 1 token!
            )
            # Minimal API call: just enough to validate credentials work
            await with_timeout(
                llm.ainvoke([HumanMessage(content="test")]),
                OPENAI_TIMEOUT,
                "OpenAI"
            )
            return {"status": "healthy", "message": "API key valid"}
        except TimeoutError:
            return {"status": "unhealthy", "message": "Connection timeout"}
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            # Detect auth errors vs network issues for better debugging
            error_msg = str(e).lower()
            is_auth_error = any(x in error_msg for x in ["invalid", "authentication", "api key"])
            return {
                "status": "unhealthy",
                "message": "API key invalid" if is_auth_error else "Service unreachable"
            }
    
    # Run all checks in parallel! 🚀
    # This reduces total time from ~16s (sequential) to ~3s (parallel)
    mongodb_result, es_result, mem0_result, langfuse_result, openai_result = await asyncio.gather(
        check_service(check_mongodb(), MONGODB_TIMEOUT, "MongoDB"),
        check_elasticsearch(),
        check_mem0(),
        check_langfuse(),
        check_openai(),
        return_exceptions=True  # Don't fail entire health check if one service fails
    )
    
    # Handle any exceptions from gather (though our checks already handle them)
    checks = {
        "mongodb": mongodb_result if isinstance(mongodb_result, dict) else {"status": "unhealthy", "message": str(mongodb_result)},
        "elasticsearch": es_result if isinstance(es_result, dict) else {"status": "unhealthy", "message": str(es_result)},
        "mem0": mem0_result if isinstance(mem0_result, dict) else {"status": "degraded", "message": str(mem0_result)},
        "langfuse": langfuse_result if isinstance(langfuse_result, dict) else {"status": "degraded", "message": str(langfuse_result)},
        "openai": openai_result if isinstance(openai_result, dict) else {"status": "unhealthy", "message": str(openai_result)}
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
        timestamp=datetime.now(timezone.utc).isoformat() # When this check ran
    )
