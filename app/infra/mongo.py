"""MongoDB client with connection pooling"""
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from app.infra.config import settings
from typing import Optional


class MongoDBClient:
    """MongoDB client with connection pooling"""
    
    def __init__(self, uri: Optional[str] = None):
        uri = uri or settings.mongodb_uri
        self.client = AsyncIOMotorClient(
            uri,
            server_api=ServerApi('1'),
            maxPoolSize=50,  # Max connections per process
            minPoolSize=10,  # Min connections maintained
            maxIdleTimeMS=30000,  # Close idle connections after 30s
            connectTimeoutMS=5000,  # 5s connection timeout
            serverSelectionTimeoutMS=5000  # 5s server selection timeout
        )
        self.db = self.client[settings.mongodb_db_name]
        
        # Collections
        self.conversations = self.db.conversations
        self.messages = self.db.messages
        self.summaries = self.db.summaries
        
        # Food delivery collections
        self.orders = self.db.orders
        self.users = self.db.users  # Renamed from customers
        self.zones = self.db.zones
        self.zone_metrics_history = self.db.zone_metrics_history
        self.restaurants = self.db.restaurants
        self.restaurant_metrics_history = self.db.restaurant_metrics_history
        self.support_tickets = self.db.support_tickets  # New - merges incidents + cases
    
    async def close(self):
        """Close MongoDB connection"""
        self.client.close()
    
    async def ping(self) -> bool:
        """Check MongoDB connection"""
        try:
            await self.client.admin.command('ping')
            return True
        except Exception:
            return False


# Global MongoDB client instance
mongodb_client: Optional[MongoDBClient] = None


async def get_mongodb_client() -> MongoDBClient:
    """Get or create MongoDB client instance"""
    global mongodb_client
    if mongodb_client is None:
        mongodb_client = MongoDBClient()
    return mongodb_client
