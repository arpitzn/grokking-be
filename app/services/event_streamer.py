"""Simple, clean event streaming for SSE with type-safe enums"""

import json
import asyncio
import hashlib
import re
from enum import Enum
from typing import Dict, Any, AsyncGenerator

from app.agent.state import EventClass
from app.utils.tool_observability import get_pending_events


class EventType(str, Enum):
    """SSE event types"""
    THINKING = "thinking"
    EVIDENCE_CARD = "evidence_card"
    EVIDENCE_GAP = "evidence_gap"
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
    
    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        self.seen_events = 0  # Single counter for unified event stream
        self.seen_evidence = {
            EvidenceSource.MONGO: 0,
            EvidenceSource.POLICY: 0,
            EvidenceSource.MEMORY: 0
        }
        self.full_response = ""
        self.phase_emitted = set()  # Track which phases already shown
        self.seen_evidence_hashes = {
            EvidenceSource.MONGO: set(),
            EvidenceSource.POLICY: set(),
            EvidenceSource.MEMORY: set()
        }  # Track seen evidence item hashes for deduplication
    
    def _should_stream(self, event_class: str) -> bool:
        """Determine if event should be streamed based on debug mode"""
        if self.debug_mode:
            return True
        return event_class in ["user", "explainability"]
    
    def _sanitize_content(self, content: str) -> str:
        """Remove unresolved template variables and sensitive patterns"""
        # Only redact actual sensitive patterns, not template variables
        sanitized = re.sub(r'(password|secret|token|api_key|credential)\s*[:=]\s*\S+', 
                          '[redacted]', content, flags=re.IGNORECASE)
        return sanitized
    
    def _format_sse(self, data: Dict[str, Any], event_class: str) -> str:
        """Format data as SSE event with classification and filtering"""
        if not self._should_stream(event_class):
            return ""
        
        data["class"] = event_class
        
        # Sanitize content fields
        if "content" in data:
            data["content"] = self._sanitize_content(data["content"])
        
        return f"data: {json.dumps(data)}\n\n"
    
    async def stream_tool_events(self) -> AsyncGenerator[str, None]:
        """Stream tool observability events (DEBUG class - hidden by default)"""
        for event in get_pending_events():
            result = self._format_sse({
                "event": EventType.TOOL_EVENT,
                "type": event["type"],
                "payload": event["payload"]
            }, EventClass.DEBUG.value)
            if result:  # Only yield if not filtered
                yield result
    
    async def stream_phase_events(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream phase-level summaries instead of verbose CoT entries.
        Only emits one summary per phase (deduplicated).
        """
        if "events" not in node_output:
            return
        
        events = node_output["events"]
        new_events = events[self.seen_events:]
        
        for event in new_events:
            phase = event.get("phase")
            event_class = event.get("class", "explainability")
            
            # Only emit each phase once
            if phase and phase not in self.phase_emitted:
                self.phase_emitted.add(phase)
                
                # Create rich phase summary
                summary = self._create_phase_summary(phase, event)
                
                result = self._format_sse({
                    "event": EventType.THINKING,
                    "phase": phase,
                    "turn": event.get("turn", 1),
                    "content": summary
                }, event_class)
                if result:  # Only yield if not filtered
                    yield result
        
        self.seen_events = len(events)
    
    def _create_phase_summary(self, phase: str, event: Dict) -> str:
        """
        Create concise, informative phase summary with metrics.
        Falls back to event content if no custom summary defined.
        """
        content = event.get("content", "")
        metadata = event.get("metadata", {})
        
        # Custom summaries with metrics
        summaries = {
            "ingestion": self._format_ingestion_summary(metadata, content),
            "intent_classification": self._format_intent_summary(metadata, content),
            "planning": f"Planning: {content}",
            "searching": self._format_search_summary(metadata),
            "reasoning": self._format_reasoning_summary(metadata),
            "generating": "Composing response...",
            "guardrails": self._format_guardrails_summary(metadata, content)
        }
        
        return summaries.get(phase, content)
    
    def _format_ingestion_summary(self, metadata: Dict, content: str) -> str:
        """Format ingestion phase summary with entity extraction details"""
        entities = metadata.get("entities", [])
        confidence = metadata.get("confidence", 0.0)
        order_id = metadata.get("order_id")
        
        # Add confidence emoji
        emoji = "✓" if confidence > 0.7 else "~" if confidence > 0.5 else "⚠️"
        
        parts = []
        if entities:
            parts.append(f"entities: {', '.join(entities)}")
        if confidence > 0:
            parts.append(f"confidence: {confidence:.2f}")
        if order_id:
            parts.append(f"order: {order_id}")
        
        if parts:
            return f"{emoji} Extracted {', '.join(parts)}"
        else:
            return f"{emoji} {content[:50]}..." if len(content) > 50 else f"{emoji} {content}"
    
    def _format_intent_summary(self, metadata: Dict, content: str) -> str:
        """Format intent classification summary with confidence"""
        confidence = metadata.get("confidence", 0.0)
        emoji = "✓" if confidence > 0.8 else "~" if confidence > 0.6 else "⚠️"
        
        sla_risk = metadata.get("SLA_risk", False)
        safety_flags = metadata.get("safety_flags", [])
        
        parts = [content]
        if confidence > 0:
            parts.append(f"confidence: {confidence:.2f}")
        if sla_risk:
            parts.append("🚨 SLA risk")
        if safety_flags:
            parts.append(f"⚠️ {len(safety_flags)} safety flags")
        
        return f"{emoji} {' | '.join(parts)}"
    
    def _format_guardrails_summary(self, metadata: Dict, content: str) -> str:
        """Format guardrails summary with routing decision and confidence"""
        confidence = metadata.get("confidence", 0.0)
        needs_more_data = metadata.get("needs_more_data", False)
        threshold = metadata.get("confidence_threshold", 0.7)
        
        # Decision emoji
        passed = confidence >= threshold
        emoji = "✓" if passed else "⚠️"
        
        parts = [content]
        if confidence > 0:
            parts.append(f"confidence: {confidence:.2f} (threshold: {threshold})")
        if needs_more_data:
            parts.append("🔍 needs more data")
        
        return f"{emoji} {' | '.join(parts)}"
    
    def _format_search_summary(self, metadata: Dict) -> str:
        """Format search phase summary with evidence counts"""
        agents = metadata.get("agents", [])
        evidence_count = metadata.get("evidence_count", 0)
        if agents and evidence_count > 0:
            return f"Retrieved {evidence_count} evidence items from {len(agents)} sources"
        elif evidence_count > 0:
            return f"Retrieved {evidence_count} evidence items"
        else:
            return "Gathering evidence from multiple sources"
    
    def _format_reasoning_summary(self, metadata: Dict) -> str:
        """Format reasoning phase summary with hypothesis count, confidence, and self-reflection"""
        hypothesis_count = metadata.get("hypothesis_count", 0)
        confidence = metadata.get("confidence", 0.0)
        evidence_quality = metadata.get("evidence_quality", "unknown")
        needs_more_data = metadata.get("needs_more_data", False)
        conflicts = metadata.get("conflicts", 0)
        overall_confidence = metadata.get("overall_confidence", confidence)
        
        # Add confidence emoji
        conf_emoji = "✓" if overall_confidence > 0.75 else "~" if overall_confidence > 0.6 else "⚠️"
        
        parts = []
        if hypothesis_count > 0:
            parts.append(f"{hypothesis_count} hypotheses")
        
        parts.append(f"confidence: {overall_confidence:.2f}")
        
        if evidence_quality != "unknown":
            quality_emoji = {"high": "✓", "medium": "~", "low": "⚠️"}.get(evidence_quality, "")
            parts.append(f"evidence: {quality_emoji} {evidence_quality}")
        
        if needs_more_data:
            parts.append("🔍 needs more data")
        
        if conflicts > 0:
            parts.append(f"{conflicts} conflicts")
        
        if parts:
            return f"{conf_emoji} Generated {', '.join(parts)}"
        else:
            return f"{conf_emoji} Analyzing evidence and generating response strategy"
    
    async def stream_evidence(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream new evidence cards"""
        if "evidence" not in node_output:
            return
        
        evidence = node_output["evidence"]
        for source in EvidenceSource:
            if source.value in evidence:
                source_list = evidence[source.value]
                new_items = source_list[self.seen_evidence[source]:]
                
                # Deduplicate evidence items by content hash
                unique_new_items = []
                for item in new_items:
                    item_str = json.dumps(item, sort_keys=True)
                    item_hash = hashlib.md5(item_str.encode()).hexdigest()
                    if item_hash not in self.seen_evidence_hashes[source]:
                        self.seen_evidence_hashes[source].add(item_hash)
                        unique_new_items.append(item)
                
                for item in unique_new_items:
                    result = self._format_sse({
                        "event": EventType.EVIDENCE_CARD,
                        "source": source.value,
                        "data": item
                    }, EventClass.EXPLAINABILITY.value)
                    if result:  # Only yield if not filtered
                        yield result
                self.seen_evidence[source] = len(source_list)
    
    async def stream_evidence_gaps(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream evidence gap alerts from reasoning or planner nodes"""
        
        # Check analysis for gaps
        if "analysis" in node_output:
            analysis = node_output["analysis"]
            gaps = analysis.get("gaps", [])
            
            if gaps and not hasattr(self, '_gaps_emitted'):
                self._gaps_emitted = True  # Emit once per stream
                
                gap_list = ", ".join(gaps) if isinstance(gaps, list) else str(gaps)
                
                result = self._format_sse({
                    "event": EventType.EVIDENCE_GAP,
                    "missing": gaps if isinstance(gaps, list) else [gaps],
                    "message": f"⚠️ Missing data: {gap_list}",
                    "impact": "May affect decision confidence"
                }, EventClass.EXPLAINABILITY.value)
                
                if result:
                    yield result
        
        # Check evidence items for gaps
        if "evidence" in node_output:
            evidence = node_output["evidence"]
            for source in ["mongo", "policy", "memory"]:
                if source in evidence:
                    items = evidence[source]
                    for item in items:
                        if "gaps" in item and item["gaps"]:
                            gap_key = f'_gap_{source}_emitted'
                            if not hasattr(self, gap_key):
                                setattr(self, gap_key, True)
                                
                                result = self._format_sse({
                                    "event": EventType.EVIDENCE_GAP,
                                    "source": source,
                                    "missing": item["gaps"],
                                    "message": f"⚠️ {source.upper()}: Missing {', '.join(item['gaps'])}"
                                }, EventClass.EXPLAINABILITY.value)
                                
                                if result:
                                    yield result
    
    async def stream_hypotheses(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream hypothesis updates from reasoning node"""
        if "analysis" not in node_output:
            return
        
        analysis = node_output["analysis"]
        seen_hypotheses = set()
        for hyp in analysis.get("hypotheses", []):
            # Create hash of hypothesis content for deduplication
            hyp_str = json.dumps(hyp, sort_keys=True)
            hyp_hash = hashlib.md5(hyp_str.encode()).hexdigest()
            
            if hyp_hash not in seen_hypotheses:
                seen_hypotheses.add(hyp_hash)
                result = self._format_sse({
                    "event": EventType.HYPOTHESIS_UPDATE,
                    "hypothesis": hyp
                }, EventClass.EXPLAINABILITY.value)
                if result:  # Only yield if not filtered
                    yield result
    
    async def stream_refund_recommendation(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream refund recommendation if present"""
        if "analysis" not in node_output:
            return
        
        analysis = node_output["analysis"]
        for action in analysis.get("action_candidates", []):
            if "refund" in action.get("action", "").lower():
                result = self._format_sse({
                    "event": EventType.REFUND_RECOMMENDATION,
                    "recommended": True,
                    "rationale": action.get("rationale", "")
                }, EventClass.EXPLAINABILITY.value)
                if result:  # Only yield if not filtered
                    yield result
    
    async def stream_incident_banner(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream incident banner if safety flags present"""
        if "intent" not in node_output:
            return
        
        intent = node_output["intent"]
        safety_flags = intent.get("safety_flags", [])
        if safety_flags:
            # Create hash of banner content for deduplication
            banner_content = json.dumps({
                "severity": intent.get("severity", "medium"),
                "flags": sorted(safety_flags)  # Sort for consistent hashing
            }, sort_keys=True)
            banner_hash = hashlib.md5(banner_content.encode()).hexdigest()
            
            # Use a simple set to track seen banners (recreated each call for simplicity)
            # In production, this could be instance-level if needed
            seen_banners = getattr(self, '_seen_banners', set())
            if banner_hash not in seen_banners:
                seen_banners.add(banner_hash)
                self._seen_banners = seen_banners
                result = self._format_sse({
                    "event": EventType.INCIDENT_BANNER,
                    "severity": intent.get("severity", "medium"),
                    "flags": safety_flags
                }, EventClass.EXPLAINABILITY.value)
                if result:  # Only yield if not filtered
                    yield result
    
    async def stream_response(self, node_output: Dict[str, Any], chunk_size: int = 10) -> AsyncGenerator[str, None]:
        """Stream final response token-by-token (USER class)"""
        if "final_response" not in node_output:
            return
        
        response_text = node_output["final_response"]
        
        # Stream tokens
        for i in range(0, len(response_text), chunk_size):
            delta = response_text[i:i + chunk_size]
            self.full_response += delta
            result = self._format_sse({"content": delta}, EventClass.USER.value)
            if result:  # Only yield if not filtered
                yield result
            await asyncio.sleep(0.05)
    
    async def stream_escalation(self, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream escalation event"""
        if "handover_packet" not in node_output:
            return
        
        handover = node_output["handover_packet"]
        result = self._format_sse({
            "event": EventType.ESCALATION,
            "escalation_id": handover.get("escalation_id"),
            "message": "Your case has been escalated to a human agent"
        }, EventClass.USER.value)
        if result:  # Only yield if not filtered
            yield result
        self.full_response = "Your case has been escalated to a human agent. You will be contacted shortly."
    
    async def stream_node(self, node_name: str, node_output: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Stream all events for a node update"""
        if not isinstance(node_output, dict):
            return
        
        # Stream phase events (replaces stream_cot_trace)
        async for event in self.stream_phase_events(node_output):
            yield event
        
        # Stream evidence cards (EXPLAINABILITY class)
        async for event in self.stream_evidence(node_output):
            yield event
        
        # Stream evidence gaps (NEW)
        async for event in self.stream_evidence_gaps(node_output):
            yield event
        
        # Stream incident banners (EXPLAINABILITY class)
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
        return self._format_sse({"status": StreamStatus.COMPLETED}, EventClass.USER.value)
    
    def done(self) -> str:
        """Return done marker"""
        return "data: [DONE]\n\n"
    
    def error(self, error: Exception) -> str:
        """Return error event"""
        return self._format_sse({
            "error": str(error),
            "status": StreamStatus.ERROR
        }, EventClass.USER.value)
