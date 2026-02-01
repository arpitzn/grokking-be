"""Centralized logging utilities for JSON-structured logging"""
import json
import time
import traceback
from typing import Any, Dict, Optional
import logging


def truncate_sensitive_data(data: Any, max_length: int = 100) -> Any:
    """
    Truncate sensitive data (like message content) to prevent logging full content.
    
    Args:
        data: Data to truncate (dict, str, or other)
        max_length: Maximum length for string values
    
    Returns:
        Truncated data
    """
    if isinstance(data, str):
        return data[:max_length] + "..." if len(data) > max_length else data
    elif isinstance(data, dict):
        truncated = {}
        for key, value in data.items():
            if key in ["message", "content", "query"]:
                truncated[key] = truncate_sensitive_data(value, max_length)
            else:
                truncated[key] = value
        return truncated
    elif isinstance(data, list):
        return [truncate_sensitive_data(item, max_length) for item in data[:5]]  # Limit list size
    else:
        return data


def log_request_start(
    logger: logging.Logger,
    method: str,
    path: str,
    user_id: str = None,
    body: Dict = None,
    query_params: Dict = None
):
    """Log incoming request with truncated body"""
    log_data = {
        "event": "request_start",
        "method": method,
        "path": path,
        "user_id": user_id,
        "query_params": query_params,
        "body": truncate_sensitive_data(body) if body else None,
        "timestamp": time.time()
    }
    logger.info(json.dumps(log_data))


def log_request_end(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    user_id: str = None,
    details: Dict = None
):
    """Log response with duration, status, and metrics"""
    log_data = {
        "event": "request_end",
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
        "user_id": user_id,
        "details": details or {}
    }
    logger.info(json.dumps(log_data))


def log_db_operation(
    logger: logging.Logger,
    operation: str,
    collection: str,
    result_count: int = None,
    expected: bool = True,
    user_id: str = None,
    filters: Dict = None
):
    """Log database operations with validation warnings"""
    log_data = {
        "event": "db_operation",
        "operation": operation,
        "collection": collection,
        "result_count": result_count,
        "user_id": user_id,
        "filters": filters
    }
    
    # Warn if expected data is missing
    if expected and result_count == 0:
        log_data["warning"] = f"Expected data not found in {collection}"
        logger.warning(json.dumps(log_data))
    else:
        logger.info(json.dumps(log_data))


def log_business_milestone(
    logger: logging.Logger,
    milestone: str,
    user_id: str = None,
    details: Dict = None
):
    """Log key business logic steps"""
    log_data = {
        "event": "business_milestone",
        "milestone": milestone,
        "user_id": user_id,
        "details": details or {},
        "timestamp": time.time()
    }
    logger.info(json.dumps(log_data))


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    error_type: str,
    context: Dict = None
):
    """Enhanced error logging with full context"""
    log_data = {
        "event": "error",
        "error_type": error_type,
        "message": str(error),
        "context": context or {},
        "stack_trace": traceback.format_exc()
    }
    logger.error(json.dumps(log_data))
