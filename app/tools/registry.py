"""Central registry of LangChain BaseTool instances for agentic retrieval subgraphs"""

from app.tools.elasticsearch.lookup_policy import LookupPolicyTool
from app.tools.elasticsearch.search_policies import SearchPoliciesTool
from app.tools.mem0.read_episodic_memory import ReadEpisodicMemoryTool
from app.tools.mem0.read_semantic_memory import ReadSemanticMemoryTool
from app.tools.mem0.read_procedural_memory import ReadProceduralMemoryTool
from app.tools.mongo.get_case_context import GetCaseContextTool
from app.tools.mongo.get_customer_ops_profile import GetCustomerOpsProfileTool
from app.tools.mongo.get_incident_signals import GetIncidentSignalsTool
from app.tools.mongo.get_order_timeline import GetOrderTimelineTool
from app.tools.mongo.get_restaurant_ops import GetRestaurantOpsTool
from app.tools.mongo.get_zone_ops_metrics import GetZoneOpsMetricsTool

# MongoDB tools
MONGO_TOOLS = [
    GetOrderTimelineTool(),
    GetCustomerOpsProfileTool(),
    GetZoneOpsMetricsTool(),
    GetIncidentSignalsTool(),
    GetRestaurantOpsTool(),
    GetCaseContextTool(),
]

# Policy/Elasticsearch tools
POLICY_TOOLS = [
    SearchPoliciesTool(),
    LookupPolicyTool(),
]

# Memory/Mem0 tools
MEMORY_TOOLS = [
    ReadEpisodicMemoryTool(),
    ReadSemanticMemoryTool(),
    ReadProceduralMemoryTool(),
]

# All tools combined (for reference)
ALL_TOOLS = MONGO_TOOLS + POLICY_TOOLS + MEMORY_TOOLS
