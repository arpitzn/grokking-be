"""Prompt loader for retrieval agent system prompts"""

import os
import yaml
from pathlib import Path
from typing import Dict

# Cache for loaded prompts
_prompts_cache: Dict[str, str] = {}


def _load_prompts() -> Dict[str, str]:
    """Load prompts from YAML file"""
    global _prompts_cache
    
    if _prompts_cache:
        return _prompts_cache
    
    # Get config directory path
    config_dir = Path(__file__).parent.parent.parent / "config"
    prompts_file = config_dir / "retrieval_prompts.yml"
    
    if not prompts_file.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")
    
    with open(prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Extract system prompts
    for agent_name, agent_config in data.items():
        if "system_prompt" in agent_config:
            _prompts_cache[agent_name] = agent_config["system_prompt"]
    
    return _prompts_cache


def get_prompt(agent_name: str) -> str:
    """
    Get system prompt for an agent.
    
    Args:
        agent_name: Name of the agent (e.g., "mongo_retrieval_agent")
        
    Returns:
        System prompt string
        
    Raises:
        KeyError: If agent_name not found
    """
    prompts = _load_prompts()
    
    if agent_name not in prompts:
        raise KeyError(f"Prompt not found for agent: {agent_name}. Available: {list(prompts.keys())}")
    
    return prompts[agent_name]
