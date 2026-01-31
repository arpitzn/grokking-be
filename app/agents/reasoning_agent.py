"""
Agent Responsibility:
- Fuses evidence from all sources
- Generates top N hypotheses with confidence scores
- Creates action candidates
- Identifies knowledge gaps
- Does NOT make routing decisions or generate final responses
"""

import json
from typing import Dict, Any, List

from app.agent.state import AgentState
from app.infra.llm import get_llm_service, get_expensive_model


async def reasoning_node(state: AgentState) -> AgentState:
    """
    Reasoning node: Fuses evidence and generates hypotheses.
    
    Input: evidence (mongo[], policy[], memory[]), intent, case
    Output: analysis (hypotheses[], action_candidates[], confidence, gaps)
    """
    evidence = state.get("evidence", {})
    intent = state.get("intent", {})
    case = state.get("case", {})
    
    # Collect all evidence
    mongo_evidence = evidence.get("mongo", [])
    policy_evidence = evidence.get("policy", [])
    memory_evidence = evidence.get("memory", [])
    
    # Build prompt for reasoning
    prompt = f"""You are a reasoning agent for a food delivery support system.

Case Context:
- Issue Type: {intent.get('issue_type', 'unknown')}
- Severity: {intent.get('severity', 'low')}
- Order ID: {case.get('order_id', 'N/A')}
- Customer ID: {case.get('customer_id', 'N/A')}

Evidence from MongoDB:
{json.dumps(mongo_evidence, indent=2)}

Evidence from Policies:
{json.dumps(policy_evidence, indent=2)}

Evidence from Memory:
{json.dumps(memory_evidence, indent=2)}

Analyze the evidence and provide:
1. Top 3 hypotheses with confidence scores (0.0 to 1.0)
2. Action candidates (what actions can be taken)
3. Overall confidence score (0.0 to 1.0)
4. Knowledge gaps (what information is missing)

Respond ONLY with valid JSON:
{{
    "hypotheses": [
        {{"hypothesis": "Customer eligible for refund", "confidence": 0.85, "evidence": ["order_timeline", "policy"]}},
        {{"hypothesis": "Delivery delay caused by restaurant", "confidence": 0.70, "evidence": ["order_timeline"]}}
    ],
    "action_candidates": [
        {{"action": "issue_refund", "confidence": 0.85, "rationale": "Policy allows refund for delays >45min"}},
        {{"action": "apologize_and_explain", "confidence": 0.90, "rationale": "Standard customer service"}}
    ],
    "confidence": 0.82,
    "gaps": ["restaurant_prep_time_details"]
}}
"""
    
    messages = [
        {"role": "system", "content": "You are a reasoning agent. Analyze evidence and generate hypotheses. Respond only with valid JSON."},
        {"role": "user", "content": prompt}
    ]
    
    # Use expensive model for reasoning
    llm_service = get_llm_service()
    llm = llm_service.get_llm_instance(
        model_name=get_expensive_model(),
        temperature=0.5
    )
    lc_messages = llm_service.convert_messages(messages)
    response = await llm.ainvoke(lc_messages)
    
    # Parse JSON response
    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        analysis_data = json.loads(content)
    except (json.JSONDecodeError, AttributeError) as e:
        # Fallback to default analysis
        analysis_data = {
            "hypotheses": [{"hypothesis": "Unable to analyze", "confidence": 0.0, "evidence": []}],
            "action_candidates": [],
            "confidence": 0.0,
            "gaps": ["reasoning_failed"]
        }
    
    # Populate analysis slice
    state["analysis"] = {
        "hypotheses": analysis_data.get("hypotheses", []),
        "action_candidates": analysis_data.get("action_candidates", []),
        "confidence": analysis_data.get("confidence", 0.0),
        "gaps": analysis_data.get("gaps", [])
    }
    
    # Add CoT trace entry
    if "cot_trace" not in state:
        state["cot_trace"] = []
    state["cot_trace"].append({
        "phase": "reasoning",
        "content": f"Generated {len(state['analysis']['hypotheses'])} hypotheses with confidence {state['analysis']['confidence']:.2f}"
    })
    
    return state
