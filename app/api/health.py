"""Health check endpoint"""
from fastapi import APIRouter
from app.models.schemas import HealthCheckResponse
from app.infra.mongo import get_mongodb_client
from app.infra.elasticsearch import get_elasticsearch_client
from app.infra.mem0 import get_mem0_client
from app.infra.langfuse import get_langfuse_manager
from app.infra.llm import get_llm_client
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Comprehensive health check for all dependencies
    
    Returns:
        - healthy: All checks passed
        - degraded: Some checks failed but core functionality works
        - unhealthy: Critical checks failed
    """
    checks = {}
    
    # MongoDB check
    try:
        mongo_client = await get_mongodb_client()
        mongo_healthy = await mongo_client.ping()
        checks["mongodb"] = {
            "status": "healthy" if mongo_healthy else "unhealthy",
            "message": "Connected" if mongo_healthy else "Connection failed"
        }
    except Exception as e:
        checks["mongodb"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Elasticsearch check
    try:
        es_client = await get_elasticsearch_client()
        health_info = await es_client.health_check()
        
        checks["elasticsearch"] = {
            "status": health_info["status"],
            "cluster_name": health_info.get("cluster_name"),
            "version": health_info.get("version"),
            "cluster_health": health_info.get("cluster_health"),
            "nodes": health_info.get("number_of_nodes"),
            "auth_configured": health_info.get("authentication_configured"),
            "message": health_info.get("error", "Connected")
        }
    except Exception as e:
        checks["elasticsearch"] = {
            "status": "unhealthy",
            "message": str(e)
        }
    
    # Mem0 check
    try:
        mem0_client = await get_mem0_client()
        mem0_healthy = await mem0_client.health_check()
        checks["mem0"] = {
            "status": "healthy" if mem0_healthy else "degraded",
            "message": "API accessible" if mem0_healthy else "API unreachable"
        }
    except Exception as e:
        checks["mem0"] = {
            "status": "degraded",
            "message": str(e)
        }
    
    # Langfuse check (simplified - just check if client exists)
    try:
        langfuse = get_langfuse_manager()
        checks["langfuse"] = {
            "status": "healthy",
            "message": "Client initialized"
        }
    except Exception as e:
        checks["langfuse"] = {
            "status": "degraded",
            "message": str(e)
        }
    
    # OpenAI check (simplified)
    try:
        llm_client = get_llm_client()
        checks["openai"] = {
            "status": "healthy",
            "message": "Client initialized"
        }
    except Exception as e:
        checks["openai"] = {
            "status": "degraded",
            "message": str(e)
        }
    
    # Determine overall status
    status = "healthy"
    if any(c["status"] == "unhealthy" for c in checks.values()):
        status = "unhealthy"
    elif any(c["status"] == "degraded" for c in checks.values()):
        status = "degraded"
    
    return HealthCheckResponse(
        status=status,
        checks=checks,
        timestamp=datetime.utcnow().isoformat()
    )
