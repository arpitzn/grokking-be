"""Memory message builder - converts context to proper Mem0 messages using LLM"""

import json
import logging
from typing import Dict, Any, List, Optional

from app.infra.llm import get_llm_service, get_cheap_model

logger = logging.getLogger(__name__)


class MemoryBuilder:
    """Builds properly formatted memory statements for Mem0"""
    
    @staticmethod
    async def build_episodic_user_memory(
        case: Dict[str, Any],
        intent: Dict[str, Any],
        outcome: str,
        evidence: Optional[Dict[str, List[Dict]]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> List[str]:
        """
        EPISODIC MEMORY: User behavior patterns (user-scoped)
        LLM generates declarative statements about user preferences and patterns
        Stored in Mem0 with user_id for personalization
        Satisfies hackathon requirement: "episodic memory - past incidents, conversations, outcomes"
        
        Args:
            case: Case dictionary with user/case info
            intent: Intent dictionary with issue classification
            outcome: Final response or escalation outcome
            evidence: Retrieved evidence from all sources
            analysis: Reasoning analysis with hypotheses
            conversation_history: Last few conversation messages
        
        Returns:
            List of declarative memory statements
        """
        try:
            # Extract key context
            issue_type = intent.get("issue_type", "unknown")
            severity = intent.get("severity", "low")
            sla_risk = intent.get("SLA_risk", False)
            
            # Summarize evidence
            evidence_summary = ""
            if evidence:
                mongo_items = evidence.get("mongo", [])
                if mongo_items:
                    order_data = mongo_items[0].get("data", {})
                    order_id = order_data.get("order_id", "N/A")
                    evidence_summary = f"Order {order_id}"
            
            # Extract hypotheses
            hypotheses_text = ""
            if analysis:
                hypotheses = analysis.get("hypotheses", [])
                if hypotheses:
                    top_hypothesis = hypotheses[0].get("description", "")[:100] if hypotheses else ""
                    hypotheses_text = f"Top hypothesis: {top_hypothesis}"
            
            # Extract conversation pattern
            conversation_summary = ""
            if conversation_history:
                recent_messages = conversation_history[-3:]
                conversation_summary = f"Recent {len(recent_messages)} messages in conversation"
            
            # Build LLM prompt
            prompt = f"""Generate 2-3 declarative memory statements about this user's behavior and preferences.

Context:
- Issue: {issue_type} (severity: {severity})
- SLA Risk: {sla_risk}
- Outcome: {outcome[:150]}
- Evidence: {evidence_summary}
- {hypotheses_text}
- {conversation_summary}

Rules:
- Use declarative facts only (no "remember this" or "I should remember")
- Focus on user patterns, preferences, and behaviors
- Be specific and actionable
- Each statement should be a complete sentence

Examples:
- "User prefers quick resolution for delivery delays"
- "User frequently reports quality issues with Italian restaurants"
- "User expressed frustration with 45-minute delay during dinner rush"

Generate 2-3 statements, one per line:"""

            # Call LLM
            llm_service = get_llm_service()
            llm = llm_service.get_llm_instance(
                model_name=get_cheap_model(),
                temperature=0.3,
                max_completion_tokens=200
            )
            
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response into statements
            statements = [
                s.strip() 
                for s in content.split('\n') 
                if s.strip() and not s.strip().startswith('-') and len(s.strip()) > 20
            ]
            
            # Limit to 3 statements
            if statements:
                return statements[:3]
            else:
                # Fallback to templates
                logger.warning("LLM returned no statements, using fallback templates")
                return MemoryBuilder._build_episodic_fallback(case, intent, outcome)
                
        except Exception as e:
            logger.error(f"LLM memory generation failed: {e}, using fallback templates")
            return MemoryBuilder._build_episodic_fallback(case, intent, outcome)
    
    @staticmethod
    def _build_episodic_fallback(
        case: Dict[str, Any],
        intent: Dict[str, Any],
        outcome: str
    ) -> List[str]:
        """Fallback template-based episodic memory generation"""
        memories = []
        
        issue_type = intent.get("issue_type", "unknown")
        severity = intent.get("severity", "low")
        
        if issue_type and issue_type != "unknown":
            memories.append(
                f"User reported {issue_type} issue with {severity} severity."
            )
        
        if outcome:
            if "refund" in outcome.lower():
                memories.append("User received refund as resolution.")
            elif "escalat" in outcome.lower():
                memories.append("User case required human escalation.")
            elif "resolved" in outcome.lower():
                memories.append("User issue was resolved successfully.")
        
        return memories
    
    @staticmethod
    async def build_semantic_app_memory(
        zone_id: Optional[str],
        restaurant_id: Optional[str],
        incident_signals: List[Dict],
        issue_type: str,
        evidence: Optional[Dict[str, List[Dict]]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        intent: Optional[Dict[str, Any]] = None,
        case: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        SEMANTIC MEMORY: Operational patterns (app-scoped)
        LLM generates declarative statements about system behavior and trends
        Stored in Mem0 without user_id for cross-user learning
        Satisfies hackathon requirement: "semantic memory - documents, FAQs, runbooks"
        
        Args:
            zone_id: Zone identifier
            restaurant_id: Restaurant identifier
            incident_signals: List of incident signal dictionaries
            issue_type: Type of issue encountered
            evidence: Retrieved evidence from all sources
            analysis: Reasoning analysis
            intent: Intent classification
            case: Case information
        
        Returns:
            List of declarative memory statements
        """
        try:
            # Extract operational context
            severity = intent.get("severity", "low") if intent else "low"
            sla_risk = intent.get("SLA_risk", False) if intent else False
            
            # Summarize evidence
            evidence_summary = ""
            if evidence:
                mongo_items = evidence.get("mongo", [])
                if mongo_items:
                    zone_metrics = mongo_items[0].get("data", {}).get("zone_metrics", {})
                    if zone_metrics:
                        evidence_summary = f"Zone metrics available"
            
            # Build LLM prompt
            prompt = f"""Generate 1-2 declarative memory statements about operational patterns and system behavior.

Context:
- Zone: {zone_id or 'N/A'}
- Restaurant: {restaurant_id or 'N/A'}
- Issue Type: {issue_type}
- Severity: {severity}
- SLA Risk: {sla_risk}
- Incident Signals: {len(incident_signals)} signals
- {evidence_summary}

Rules:
- Use declarative facts only (no "remember this")
- Focus on operational patterns, trends, and system behavior
- Be specific with numbers/metrics if available
- Each statement should be a complete sentence

Examples:
- "Zone B shows 40% higher rider shortages on Friday evenings (6-9pm)"
- "Restaurant X delays increase during heavy rain"
- "Zone Y experiences peak demand during dinner hours"

Generate 1-2 statements, one per line:"""

            # Call LLM
            llm_service = get_llm_service()
            llm = llm_service.get_llm_instance(
                model_name=get_cheap_model(),
                temperature=0.3,
                max_completion_tokens=150
            )
            
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            statements = [
                s.strip() 
                for s in content.split('\n') 
                if s.strip() and not s.strip().startswith('-') and len(s.strip()) > 20
            ]
            
            if statements:
                return statements[:2]
            else:
                logger.warning("LLM returned no statements, using fallback templates")
                return MemoryBuilder._build_semantic_fallback(zone_id, restaurant_id, incident_signals, issue_type)
                
        except Exception as e:
            logger.error(f"LLM semantic memory generation failed: {e}, using fallback templates")
            return MemoryBuilder._build_semantic_fallback(zone_id, restaurant_id, incident_signals, issue_type)
    
    @staticmethod
    def _build_semantic_fallback(
        zone_id: Optional[str],
        restaurant_id: Optional[str],
        incident_signals: List[Dict],
        issue_type: str
    ) -> List[str]:
        """Fallback template-based semantic memory generation"""
        memories = []
        
        if zone_id and issue_type:
            memories.append(f"Zone {zone_id} experienced {issue_type} incidents.")
        
        if restaurant_id and issue_type:
            memories.append(f"Restaurant {restaurant_id} had {issue_type} issue.")
        
        if incident_signals:
            signal_types = [sig.get("type", "") for sig in incident_signals if sig.get("type")]
            if signal_types:
                unique_types = list(set(signal_types))
                memories.append(f"Zone {zone_id or 'unknown'} shows patterns: {', '.join(unique_types)}.")
        
        return memories
    
    @staticmethod
    async def build_procedural_app_memory(
        issue_type: str,
        resolution_action: str,
        confidence: float,
        analysis: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, List[Dict]]] = None,
        intent: Optional[Dict[str, Any]] = None,
        guardrails: Optional[Dict[str, Any]] = None,
        final_response: Optional[str] = None
    ) -> List[str]:
        """
        PROCEDURAL MEMORY: "What works" knowledge (app-scoped)
        LLM generates declarative statements about effective resolution strategies
        Only stored when confidence > 0.8 to ensure quality
        Satisfies hackathon requirement: "procedural memory - what works, when, and why"
        
        Args:
            issue_type: Type of issue that was resolved
            resolution_action: Action that resolved it
            confidence: Confidence score (0.0-1.0)
            analysis: Reasoning analysis with action candidates
            evidence: Retrieved evidence
            intent: Intent classification
            guardrails: Guardrails results
            final_response: Actual response text
        
        Returns:
            List of declarative memory statements
        """
        try:
            # Only generate if confidence is high
            if confidence < 0.8:
                return []
            
            # Extract action candidates
            action_candidates_text = ""
            if analysis:
                candidates = analysis.get("action_candidates", [])
                if candidates:
                    alternatives = [c.get("action", "") for c in candidates[:2]]
                    action_candidates_text = f"Alternatives considered: {', '.join(alternatives)}"
            
            # Build LLM prompt
            prompt = f"""Generate 1 declarative memory statement about what works for resolving issues.

Context:
- Issue Type: {issue_type}
- Resolution Action: {resolution_action}
- Confidence: {confidence:.2f}
- {action_candidates_text}

Rules:
- Use declarative facts only (no "remember this")
- Focus on what action works and when
- Be specific about conditions
- Single complete sentence

Examples:
- "Offering 20% partial refund resolves most moderate delivery delays (30-60min) for high LTV customers"
- "Escalating outages to ops early reduces repeat tickets"
- "Providing clear ETA updates prevents escalation for delivery delays"

Generate 1 statement:"""

            # Call LLM
            llm_service = get_llm_service()
            llm = llm_service.get_llm_instance(
                model_name=get_cheap_model(),
                temperature=0.3,
                max_completion_tokens=100
            )
            
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            statements = [
                s.strip() 
                for s in content.split('\n') 
                if s.strip() and not s.strip().startswith('-') and len(s.strip()) > 30
            ]
            
            if statements:
                return statements[:1]
            else:
                logger.warning("LLM returned no statements, using fallback templates")
                return MemoryBuilder._build_procedural_fallback(issue_type, resolution_action, confidence)
                
        except Exception as e:
            logger.error(f"LLM procedural memory generation failed: {e}, using fallback templates")
            return MemoryBuilder._build_procedural_fallback(issue_type, resolution_action, confidence)
    
    @staticmethod
    def _build_procedural_fallback(
        issue_type: str,
        resolution_action: str,
        confidence: float
    ) -> List[str]:
        """Fallback template-based procedural memory generation"""
        memories = []
        
        if confidence > 0.8 and issue_type and resolution_action:
            memories.append(f"Offering {resolution_action} resolves most {issue_type} issues.")
        
        return memories
