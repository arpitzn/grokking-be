"""Simple, clean event streaming for SSE with type-safe enums"""

import json
import asyncio
from enum import Enum
from typing import Dict, Any, AsyncGenerator

from app.utils.tool_observability import get_pending_events


class EventType(str, Enum):
    """SSE event types"""
    THINKING = "thinking"
    EVIDENCE_CARD = "evidence_card"
    HYPOTHESIS_UPDATE = "hypothesis_update"
    REFUND_RECOMMENDATION = "refund_recommendation"
    INCIDENT_BANNER = "incident_banner"
    TOOL_EVENT = "tool_event"
    ESCALATION = "escalation"
    CONTENT = "content"
    STATUS = "status"


class ThinkingPhase(str, Enum):
    """Thinking phases for CoT trace"""
    INGESTION = "ingestion"
    INTENT_CLASSIFICATION = "intent_classification"
    PLANNING = "planning"
    SEARCHING = "searching"
    REASONING = "reasoning"
    GENERATING = "generating"
    GUARDRAILS = "guardrails"


class EvidenceSource(str, Enum):
    """Evidence sources"""
    MONGO = "mongo"
    POLICY = "policy"
    MEMORY = "memory"


class StreamStatus(str, Enum):
    """Stream status values"""
    COMPLETED = "completed"
    ERROR = "error"


class EventStreamer:
    """Simple, clean event streaming for SSE"""
    
    def __init__(self):
        self.seen_cot = 0
        self.seen_evidence = {
            EvidenceSource.MONGO: 0,
            EvidenceSource.POLICY: 0,
            EvidenceSource.MEMORY: 0
        }
        self.full_response = ""
    
    def _format_sse(self, data: Dict[str, Any]) -> str:
        """Format data as SSE event"""
        return f"data: {json.dumps(data)}\n\n"
    
    async def stream_tool_events(self) -> AsyncGenerator[str, None]:
        """Stream tool observability events"""
        for event in get_pending_events():
            yield self._format_sse({
                "event": EventType.TOOL_EVENT,
                "type": event["type"],
                "payload": event["payload"]
            })
    
    async def stream_cot_trace(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream new CoT trace entries"""
        if "cot_trace" not in node_output:
            return
        
        cot_trace = node_output["cot_trace"]
        for entry in cot_trace[self.seen_cot:]:
            yield self._format_sse({
                "event": EventType.THINKING,
                "phase": entry.get("phase"),
                "turn": entry.get("turn", 1),
                "content": entry.get("content")
            })
        self.seen_cot = len(cot_trace)
    
    async def stream_evidence(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream new evidence cards"""
        if "evidence" not in node_output:
            return
        
        evidence = node_output["evidence"]
        for source in EvidenceSource:
            if source.value in evidence:
                new_items = evidence[source.value][self.seen_evidence[source]:]
                for item in new_items:
                    yield self._format_sse({
                        "event": EventType.EVIDENCE_CARD,
                        "source": source.value,
                        "data": item
                    })
                self.seen_evidence[source] = len(evidence[source.value])
    
    async def stream_hypotheses(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream hypothesis updates from reasoning node"""
        if "analysis" not in node_output:
            return
        
        analysis = node_output["analysis"]
        for hyp in analysis.get("hypotheses", []):
            yield self._format_sse({
                "event": EventType.HYPOTHESIS_UPDATE,
                "hypothesis": hyp
            })
    
    async def stream_refund_recommendation(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream refund recommendation if present"""
        if "analysis" not in node_output:
            return
        
        analysis = node_output["analysis"]
        for action in analysis.get("action_candidates", []):
            if "refund" in action.get("action", "").lower():
                yield self._format_sse({
                    "event": EventType.REFUND_RECOMMENDATION,
                    "recommended": True,
                    "rationale": action.get("rationale", "")
                })
    
    async def stream_incident_banner(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream incident banner if safety flags present"""
        if "intent" not in node_output:
            return
        
        intent = node_output["intent"]
        safety_flags = intent.get("safety_flags", [])
        if safety_flags:
            yield self._format_sse({
                "event": EventType.INCIDENT_BANNER,
                "severity": intent.get("severity", "medium"),
                "flags": safety_flags
            })
    
    async def stream_response(self, node_output: Dict[str, Any], chunk_size: int = 10) -> AsyncGenerator[str, None]:
        """Stream final response token-by-token"""
        if "final_response" not in node_output:
            return
        
        response_text = node_output["final_response"]
        
        # Thinking phase
        yield self._format_sse({
            "event": EventType.THINKING,
            "phase": ThinkingPhase.GENERATING,
            "content": "Composing response..."
        })
        
        # Stream tokens
        for i in range(0, len(response_text), chunk_size):
            delta = response_text[i:i + chunk_size]
            self.full_response += delta
            yield self._format_sse({"content": delta})
            await asyncio.sleep(0.05)
    
    async def stream_escalation(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream escalation event"""
        if "handover_packet" not in node_output:
            return
        
        handover = node_output["handover_packet"]
        yield self._format_sse({
            "event": EventType.ESCALATION,
            "escalation_id": handover.get("escalation_id"),
            "message": "Your case has been escalated to a human agent"
        })
        self.full_response = "Your case has been escalated to a human agent. You will be contacted shortly."
    
    async def stream_node(self, node_name: str, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream all events for a node update"""
        if not isinstance(node_output, dict):
            return
        
        # Always stream CoT and evidence
        async for event in self.stream_cot_trace(node_output):
            yield event
        async for event in self.stream_evidence(node_output):
            yield event
        async for event in self.stream_incident_banner(node_output):
            yield event
        
        # Node-specific events
        if node_name == "reasoning":
            async for event in self.stream_hypotheses(node_output):
                yield event
            async for event in self.stream_refund_recommendation(node_output):
                yield event
        
        elif node_name == "response_synthesis":
            async for event in self.stream_response(node_output):
                yield event
        
        elif node_name == "human_escalation":
            async for event in self.stream_escalation(node_output):
                yield event
    
    def completion(self) -> str:
        """Return completion event"""
        return self._format_sse({"status": StreamStatus.COMPLETED})
    
    def done(self) -> str:
        """Return done marker"""
        return "data: [DONE]\n\n"
    
    def error(self, error: Exception) -> str:
        """Return error event"""
        return self._format_sse({
            "error": str(error),
            "status": StreamStatus.ERROR
        })
