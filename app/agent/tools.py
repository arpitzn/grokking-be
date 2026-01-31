# """Tool implementations for internal RAG and external search"""
# from app.services.knowledge import retrieve_chunks
# from app.infra.external_search import external_search_stub
# from typing import Dict, Any
# import logging

# logger = logging.getLogger(__name__)


# async def internal_rag_tool(user_id: str, query: str, top_k: int = 3) -> Dict[str, Any]:
#     """
#     Internal RAG tool - retrieves chunks from Elasticsearch
    
#     Returns:
#         {
#             "source": "internal",
#             "chunks": [...],
#             "query": "..."
#         }
#     """
#     try:
#         chunks = await retrieve_chunks(user_id=user_id, query=query, top_k=top_k)
#         return {
#             "source": "internal",
#             "chunks": chunks,
#             "query": query
#         }
#     except Exception as e:
#         logger.error(f"Internal RAG tool error: {e}")
#         return {
#             "source": "internal",
#             "chunks": [],
#             "query": query,
#             "error": str(e)
#         }


# async def external_search_tool(query: str, max_results: int = 3) -> Dict[str, Any]:
#     """
#     External search tool - uses stub for demo
    
#     Returns:
#         {
#             "source": "external",
#             "results": [...],
#             "query": "...",
#             "is_stub": true
#         }
#     """
#     try:
#         search_results = await external_search_stub.search(query=query, max_results=max_results)
#         return {
#             "source": "external",
#             "results": search_results.get("results", []),
#             "query": query,
#             "is_stub": search_results.get("is_stub", True)
#         }
#     except Exception as e:
#         logger.error(f"External search tool error: {e}")
#         return {
#             "source": "external",
#             "results": [],
#             "query": query,
#             "error": str(e)
#         }
