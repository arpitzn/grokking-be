"""Mock data generation utility for food delivery domain testing"""

from datetime import datetime, timedelta
from typing import Dict, Any, List
import random


def generate_mock_order_timeline(order_id: str) -> Dict[str, Any]:
    """Generate mock order timeline data"""
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    return {
        "order_id": order_id,
        "status": random.choice(["delivered", "in_transit", "preparing", "pending"]),
        "created_at": (base_time - timedelta(hours=2)).isoformat(),
        "events": [
            {
                "timestamp": (base_time - timedelta(hours=2)).isoformat(),
                "event": "order_placed",
                "status": "pending"
            },
            {
                "timestamp": (base_time - timedelta(hours=1, minutes=45)).isoformat(),
                "event": "restaurant_confirmed",
                "status": "confirmed"
            },
            {
                "timestamp": (base_time - timedelta(hours=1, minutes=30)).isoformat(),
                "event": "picked_up",
                "status": "in_transit"
            },
            {
                "timestamp": (base_time - timedelta(minutes=15)).isoformat(),
                "event": "delivered",
                "status": "delivered"
            }
        ],
        "estimated_delivery": (base_time - timedelta(minutes=30)).isoformat(),
        "actual_delivery": (base_time - timedelta(minutes=15)).isoformat(),
        "delivery_delay_minutes": random.randint(0, 30)
    }


def generate_mock_customer_profile(customer_id: str) -> Dict[str, Any]:
    """Generate mock customer operations profile"""
    return {
        "customer_id": customer_id,
        "total_orders": random.randint(10, 100),
        "lifetime_value": round(random.uniform(200, 2000), 2),
        "avg_order_value": round(random.uniform(20, 50), 2),
        "refund_count": random.randint(0, 5),
        "refund_rate": round(random.uniform(0, 0.1), 3),
        "last_order_date": (datetime.utcnow() - timedelta(days=random.randint(0, 30))).isoformat(),
        "preferred_cuisines": random.sample(["Italian", "Chinese", "Mexican", "Indian", "Thai"], k=3),
        "avg_rating_given": round(random.uniform(3.5, 5.0), 1),
        "complaint_count": random.randint(0, 3),
        "vip_status": random.choice([True, False])
    }


def generate_mock_zone_metrics(zone_id: str) -> Dict[str, Any]:
    """Generate mock zone operations metrics"""
    return {
        "zone_id": zone_id,
        "time_window": "24h",
        "total_orders": random.randint(500, 2000),
        "avg_delivery_time_minutes": random.randint(25, 45),
        "on_time_delivery_rate": round(random.uniform(0.75, 0.95), 2),
        "incident_count": random.randint(5, 30),
        "incident_rate": round(random.uniform(0.005, 0.02), 3),
        "active_drivers": random.randint(20, 60),
        "avg_restaurant_prep_time": random.randint(15, 25),
        "weather_alert": random.choice([True, False]),
        "traffic_alert": random.choice([True, False])
    }


def generate_mock_policy_results(query: str) -> Dict[str, Any]:
    """Generate mock policy search results"""
    return {
        "query": query,
        "results": [
            {
                "policy_id": "POL-REFUND-001",
                "title": "Refund Policy - Food Quality Issues",
                "content": "Customers are eligible for a full refund if food quality issues are reported within 2 hours of delivery...",
                "relevance_score": 0.92,
                "document_type": "policy",
                "section": "refunds",
                "effective_date": "2025-01-01"
            },
            {
                "policy_id": "POL-REFUND-002",
                "title": "Refund Policy - Delivery Delays",
                "content": "For delivery delays exceeding 45 minutes, customers may request a partial refund...",
                "relevance_score": 0.85,
                "document_type": "policy",
                "section": "refunds",
                "effective_date": "2025-01-01"
            }
        ],
        "total_results": 2
    }


def generate_mock_memory_results(user_id: str, query: str) -> Dict[str, Any]:
    """Generate mock memory (episodic) results"""
    return {
        "user_id": user_id,
        "query": query,
        "memories": [
            {
                "memory_id": "mem_001",
                "content": "Customer previously reported food quality issue with Italian restaurant. Refund was issued.",
                "timestamp": (datetime.utcnow() - timedelta(days=15)).isoformat(),
                "similarity_score": 0.88
            },
            {
                "memory_id": "mem_002",
                "content": "Customer had delivery delay complaint last month. Partial refund provided.",
                "timestamp": (datetime.utcnow() - timedelta(days=10)).isoformat(),
                "similarity_score": 0.75
            }
        ],
        "total_found": 2
    }
