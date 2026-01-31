# """
# Agent Responsibility:
# - Calls Elasticsearch policy tools
# - Retrieves relevant policies/SOPs/SLAs
# - Records execution status in retrieval_status
# - Populates evidence.policy[]
# - Does NOT reason or generate responses
# """

# from datetime import datetime, timezone
# from typing import Dict, Any

# from app.agent.state import AgentState
# from app.models.tool_spec import ToolCriticality
# from app.tools.elasticsearch.lookup_policy import lookup_policy, TOOL_SPEC as LOOKUP_POLICY_SPEC
# from app.tools.elasticsearch.search_policies import search_policies, TOOL_SPEC as SEARCH_POLICIES_SPEC


# # Tool registry for Elasticsearch tools
# TOOL_REGISTRY = {
#     "search_policies": (search_policies, SEARCH_POLICIES_SPEC),
#     "lookup_policy": (lookup_policy, LOOKUP_POLICY_SPEC),
# }


# async def policy_rag_node(state: AgentState) -> AgentState:
#     """
#     Policy RAG node: Calls Elasticsearch policy tools.
    
#     Input: plan.tool_selection, intent, case
#     Output: evidence.policy[], retrieval_status.policy
#     """
#     plan = state.get("plan", {})
#     tool_selection = plan.get("tool_selection", [])
#     intent = state.get("intent", {})
#     case = state.get("case", {})
    
#     # Filter to only Elasticsearch tools
#     policy_tools = [tool for tool in tool_selection if tool in TOOL_REGISTRY]
    
#     # Initialize evidence and status
#     if "evidence" not in state:
#         state["evidence"] = {}
#     if "policy" not in state["evidence"]:
#         state["evidence"]["policy"] = []
    
#     if "retrieval_status" not in state:
#         state["retrieval_status"] = {}
    
#     failed_tools = []
    
#     # Build search query from intent
#     issue_type = intent.get("issue_type", "other")
#     search_query = f"{issue_type} policy refund delivery"
    
#     # Call each selected Elasticsearch tool
#     for tool_name in policy_tools:
#         if tool_name not in TOOL_REGISTRY:
#             continue
        
#         tool_func, tool_spec = TOOL_REGISTRY[tool_name]
        
#         try:
#             # Prepare tool arguments
#             if tool_name == "search_policies":
#                 result = await tool_func(
#                     query=search_query,
#                     filters={"issue_type": issue_type},
#                     top_k=5
#                 )
#             elif tool_name == "lookup_policy":
#                 # Would need policy_id from previous search or case
#                 result = await tool_func(
#                     doc_id=f"POL-{issue_type.upper()}-001",
#                     section_id=None
#                 )
#             else:
#                 continue
            
#             # Handle failures based on ToolSpec criticality
#             if result.tool_result.status.value == "failed":
#                 if tool_spec.criticality == ToolCriticality.SAFETY_CRITICAL:
#                     failed_tools.append(tool_name)
#                 elif tool_spec.criticality == ToolCriticality.DECISION_CRITICAL:
#                     failed_tools.append(tool_name)
#                     state["evidence"]["policy"].append(result.dict())
#                 elif tool_spec.criticality == ToolCriticality.NON_CRITICAL:
#                     state["evidence"]["policy"].append(result.dict())
#             else:
#                 # Success - add to evidence
#                 state["evidence"]["policy"].append(result.dict())
        
#         except Exception as e:
#             failed_tools.append(tool_name)
#             from app.models.evidence import ToolResult, ToolStatus
#             failed_envelope = {
#                 "source": "elasticsearch",
#                 "entity_refs": [],
#                 "freshness": datetime.now(timezone.utc).isoformat(),
#                 "confidence": 0.0,
#                 "data": {},
#                 "gaps": [f"{tool_name}_exception"],
#                 "provenance": {"tool": tool_name, "error": str(e)},
#                 "tool_result": {"status": "failed", "error": str(e)}
#             }
#             state["evidence"]["policy"].append(failed_envelope)
    
#     # Update retrieval status
#     state["retrieval_status"]["policy"] = {
#         "completed": True,
#         "failed_tools": failed_tools,
#         "successful_tools": [tool for tool in policy_tools if tool not in failed_tools]
#     }
    
#     return state
