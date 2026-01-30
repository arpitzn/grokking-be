"""Chat streaming endpoint for food delivery domain"""

import asyncio
import json
import logging
import time

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.infra.guardrails import get_guardrails_manager
from app.infra.langfuse_callback import langfuse_handler, langfuse
from app.models.schemas import CaseRequest
from app.services.conversation import create_conversation, insert_message
from app.utils.logging_utils import (
    log_business_milestone,
    log_error_with_context,
    log_request_start,
)
from app.utils.tool_observability import get_pending_events
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/stream")
async def chat_stream(request: CaseRequest):
    """
    Main streaming endpoint for food delivery chat
    
    Returns SSE stream with token-by-token response and custom UI events.
    
    Custom events:
    - thinking: CoT trace updates
    - evidence_card: Evidence envelope data
    - refund_recommendation: Refund decision
    - incident_banner: Incident alerts
    - hypothesis_update: Reasoning hypotheses
    - tool_event: Tool observability events
    """
    start_time = time.time()

    # Log request start
    log_request_start(
        logger,
        "POST",
        "/chat/stream",
        user_id=request.user_id,
        body={
            "message": request.message[:100],
            "conversation_id": request.conversation_id,
            "persona": request.persona,
            "channel": request.channel,
        },
    )

    # Get guardrails manager (singleton, initialized at startup)
    guardrails = get_guardrails_manager()

    # Validate input with NeMo Guardrails
    validation_result = await guardrails.validate_input(
        request.message, request.user_id
    )

    log_business_milestone(
        logger,
        "guardrails_validation_complete",
        user_id=request.user_id,
        details={
            "passed": validation_result.passed,
            "detection_type": validation_result.detection_type,
        },
    )

    # If guardrails detected an issue, return a friendly streaming response
    if not validation_result.passed:
        log_business_milestone(
            logger,
            "guardrails_friendly_response",
            user_id=request.user_id,
            details={
                "detection_type": validation_result.detection_type,
                "duration_ms": (time.time() - start_time) * 1000,
            },
        )

        async def guardrail_response_generator():
            """Stream a friendly guardrail message to the user"""
            friendly_message = validation_result.message
            chunk_size = 10
            for i in range(0, len(friendly_message), chunk_size):
                delta = friendly_message[i : i + chunk_size]
                yield f"data: {json.dumps({'content': delta})}\n\n"

            yield f"data: {json.dumps({'status': 'completed', 'guardrail_triggered': validation_result.detection_type})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            guardrail_response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Message passed guardrails - will be processed by agent system
    logger.info("Message passed guardrails, starting agent system")

    # Create conversation if needed
    conversation_id = request.conversation_id
    if not conversation_id:
        log_business_milestone(
            logger, "conversation_creation_start", user_id=request.user_id
        )
        conversation_id = await create_conversation(request.user_id)
        log_business_milestone(
            logger,
            "conversation_created",
            user_id=request.user_id,
            details={"conversation_id": conversation_id},
        )

    # Insert user message
    user_message_id = await insert_message(
        conversation_id=conversation_id, role="user", content=request.message
    )

    async def event_generator():
        """Generate SSE events with Chain-of-Thought streaming and custom UI events"""
        try:
            # Initialize state with new AgentState structure
            initial_state: AgentState = {
                "case": {
                    "persona": request.persona or "customer",
                    "channel": request.channel or "web",
                    "raw_text": request.message,
                    "user_id": request.user_id,
                    "conversation_id": conversation_id,
                    "order_id": None,
                    "customer_id": request.user_id,
                    "zone_id": None,
                    "restaurant_id": None,
                    "locale": "en-US",
                },
                "intent": {},
                "plan": {},
                "evidence": {},
                "retrieval_status": {},
                "analysis": {},
                "guardrails": {},
                "final_response": "",
                "handover_packet": None,
                "working_memory": [],
                "conversation_summary": None,
                "trace_events": [],
                "cot_trace": [],
            }

            # Build LangGraph config with callbacks and metadata
            langfuse_config = {
                "callbacks": [langfuse_handler],
                "metadata": {
                    "langfuse_user_id": request.user_id,
                    "langfuse_session_id": conversation_id,
                    "message_preview": request.message[:100],
                    "persona": request.persona,
                    "channel": request.channel,
                },
            }

            # Get graph
            graph = get_graph()

            # Track state for streaming
            seen_cot_count = 0
            full_response = ""
            last_evidence_count = {"mongo": 0, "policy": 0, "memory": 0}

            # Stream graph execution with callbacks
            async for chunk in graph.astream(initial_state, config=langfuse_config):
                # Stream tool observability events
                tool_events = get_pending_events()
                for event in tool_events:
                    yield f"data: {json.dumps({'event': 'tool_event', 'type': event['type'], 'payload': event['payload']})}\n\n"

                if isinstance(chunk, dict):
                    # LangGraph returns state updates per node
                    for node_name, node_output in chunk.items():
                        # Log graph node execution
                        log_business_milestone(
                            logger,
                            f"graph_node_{node_name}",
                            user_id=request.user_id,
                            details={"conversation_id": conversation_id},
                        )

                        if not isinstance(node_output, dict):
                            continue

                        # Stream CoT trace entries
                        if "cot_trace" in node_output:
                            cot_trace = node_output["cot_trace"]
                            for entry in cot_trace[seen_cot_count:]:
                                yield f"data: {json.dumps({'event': 'thinking', 'phase': entry['phase'], 'content': entry['content']})}\n\n"
                            seen_cot_count = len(cot_trace)

                        # Stream evidence cards as they appear
                        if "evidence" in node_output:
                            evidence = node_output["evidence"]
                            for source in ["mongo", "policy", "memory"]:
                                if source in evidence:
                                    new_evidence = evidence[source][last_evidence_count[source]:]
                                    for ev in new_evidence:
                                        yield f"data: {json.dumps({'event': 'evidence_card', 'source': source, 'data': ev})}\n\n"
                                    last_evidence_count[source] = len(evidence[source])

                        # Stream hypothesis updates from reasoning
                        if node_name == "reasoning" and "analysis" in node_output:
                            analysis = node_output["analysis"]
                            hypotheses = analysis.get("hypotheses", [])
                            for hyp in hypotheses:
                                yield f"data: {json.dumps({'event': 'hypothesis_update', 'hypothesis': hyp})}\n\n"

                        # Stream refund recommendation if applicable
                        if node_name == "reasoning" and "analysis" in node_output:
                            analysis = node_output["analysis"]
                            action_candidates = analysis.get("action_candidates", [])
                            for action in action_candidates:
                                if "refund" in action.get("action", "").lower():
                                    yield f"data: {json.dumps({'event': 'refund_recommendation', 'recommended': True, 'rationale': action.get('rationale', '')})}\n\n"

                        # Stream incident banner if safety flags present
                        if "intent" in node_output:
                            intent = node_output["intent"]
                            safety_flags = intent.get("safety_flags", [])
                            if safety_flags:
                                yield f"data: {json.dumps({'event': 'incident_banner', 'severity': intent.get('severity', 'medium'), 'flags': safety_flags})}\n\n"

                        # Stream final response from response_synthesis
                        if node_name == "response_synthesis" and "final_response" in node_output:
                            response_text = node_output["final_response"]

                            log_business_milestone(
                                logger,
                                "response_generation_complete",
                                user_id=request.user_id,
                                details={
                                    "conversation_id": conversation_id,
                                    "response_length": len(response_text),
                                    "duration_ms": (time.time() - start_time) * 1000,
                                },
                            )

                            yield f"data: {json.dumps({'event': 'thinking', 'phase': 'generating', 'content': 'Composing response...'})}\n\n"

                            # Stream response token-by-token
                            chunk_size = 10
                            for i in range(0, len(response_text), chunk_size):
                                delta = response_text[i : i + chunk_size]
                                full_response += delta
                                yield f"data: {json.dumps({'content': delta})}\n\n"
                                await asyncio.sleep(0.05)

                        # Stream handover packet from human_escalation
                        if node_name == "human_escalation" and "handover_packet" in node_output:
                            handover = node_output["handover_packet"]
                            yield f"data: {json.dumps({'event': 'escalation', 'escalation_id': handover.get('escalation_id'), 'message': 'Your case has been escalated to a human agent'})}\n\n"
                            full_response = "Your case has been escalated to a human agent. You will be contacted shortly."

            # Insert assistant message
            if full_response:
                await insert_message(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=full_response,
                )

            # Send completion event
            log_business_milestone(
                logger,
                "stream_completed",
                user_id=request.user_id,
                details={
                    "conversation_id": conversation_id,
                    "total_duration_ms": (time.time() - start_time) * 1000,
                    "response_length": len(full_response),
                },
            )
            yield f"data: {json.dumps({'status': 'completed'})}\n\n"

        except Exception as e:
            log_error_with_context(
                logger,
                e,
                "streaming_error",
                context={
                    "user_id": request.user_id,
                    "conversation_id": conversation_id,
                    "message": request.message[:100],
                    "duration_ms": (time.time() - start_time) * 1000,
                },
            )
            error_data = json.dumps({"error": str(e), "status": "error"})
            yield f"data: {error_data}\n\n"
        finally:
            # Ensure stream is closed
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
