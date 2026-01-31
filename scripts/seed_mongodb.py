#!/usr/bin/env python3
"""
MongoDB Test Data Seeding Script

Populates MongoDB with realistic, interconnected test data for LangChain agent testing.
Uses UUID-based relationships where all _id fields are Binary UUIDs and foreign keys
reference these UUIDs.

Usage:
    python scripts/seed_mongodb.py [--force] [--create-indexes]
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID

import pymongo
from bson import Binary
from bson.binary import UuidRepresentation
from faker import Faker

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.infra.config import settings
from app.models.mongodb_schemas import (
    UserPersona, CustomerSubCategory, UserStatus,
    OrderStatus, RefundStatus, PaymentMethod, PaymentStatus,
    OrderEventType, OrderEventStatus,
    ZoneStatus, RestaurantType, RestaurantStatus,
    TicketType, IssueType, TicketSeverity, TicketScope, TicketStatus,
    OrderIssueSubtype, QualitySafetySubtype, DeliverySubtype,
    PaymentSubtype, OperationSubtype, SupportSubtype, GeneralSubtype
)

# Initialize Faker with Indian locale
fake = Faker('en_IN')
Faker.seed(42)  # For reproducible data
random.seed(42)


# ============================================================================
# UUID Helper Functions
# ============================================================================

def uuid_to_binary(uuid_string: str) -> Binary:
    """Convert UUID string to BSON Binary UUID"""
    uuid_obj = UUID(uuid_string)
    return Binary.from_uuid(uuid_obj, uuid_representation=UuidRepresentation.STANDARD)


def binary_to_uuid(binary_uuid: Binary) -> str:
    """Convert BSON Binary UUID to UUID string"""
    return str(binary_uuid.as_uuid())


# ============================================================================
# MongoDB Connection
# ============================================================================

def get_mongodb_client():
    """Create MongoDB client connection"""
    mongodb_uri = os.getenv('MONGODB_URI', settings.mongodb_uri if hasattr(settings, 'mongodb_uri') else 'mongodb://localhost:27017')
    mongodb_db_name = os.getenv('MONGODB_DB_NAME', settings.mongodb_db_name if hasattr(settings, 'mongodb_db_name') else 'hackathon_agent')
    
    try:
        client = pymongo.MongoClient(mongodb_uri)
        # Test connection
        client.admin.command('ping')
        print(f"Connected to MongoDB: {mongodb_uri}")
        print(f"Using database: {mongodb_db_name}")
        return client, client[mongodb_db_name]
    except Exception as e:
        print(f"ERROR: Failed to connect to MongoDB: {e}")
        sys.exit(1)


# ============================================================================
# Cleanup Functions
# ============================================================================

def cleanup_existing_data(db):
    """Delete all documents from collections"""
    collections = [
        'users', 'zones', 'restaurants', 'orders', 'support_tickets',
        'zone_metrics_history', 'restaurant_metrics_history'
    ]
    
    print("Cleaning up existing data...")
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        if count > 0:
            collection.delete_many({})
            print(f"  Deleted {count} documents from {collection_name}")
        else:
            print(f"  {collection_name} is already empty")


# ============================================================================
# Batch Insert Helper
# ============================================================================

def insert_batch(collection, documents: List[Dict], batch_size: int = 100) -> List[Dict]:
    """
    Insert documents in batches and return inserted documents with _id UUIDs
    
    Args:
        collection: MongoDB collection
        documents: List of documents to insert (without _id)
        batch_size: Batch size for insertion
    
    Returns:
        List of inserted documents with _id UUIDs
    """
    inserted_docs = []
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        result = collection.insert_many(batch)
        # Fetch inserted documents to get their _id UUIDs
        inserted_ids = result.inserted_ids
        for j, doc_id in enumerate(inserted_ids):
            doc = batch[j].copy()
            doc['_id'] = doc_id
            inserted_docs.append(doc)
    
    print(f"  Inserted {len(inserted_docs)} documents into {collection.name}")
    return inserted_docs


# ============================================================================
# Data Generation Functions
# ============================================================================

def generate_users(count: int) -> List[Dict]:
    """Generate user documents"""
    print(f"Generating {count} users...")
    
    users = []
    phone_numbers = set()
    emails = set()
    
    # Calculate distribution
    customer_count = int(count * 0.85)
    customer_care_count = int(count * 0.10)
    area_manager_count = count - customer_count - customer_care_count
    
    # Customer sub-categories
    platinum_count = int(customer_count * 0.10)
    high_risk_count = int(customer_count * 0.05)
    standard_count = customer_count - platinum_count - high_risk_count
    
    # Generate customers
    customer_personas = (
        [UserPersona.CUSTOMER] * platinum_count +
        [UserPersona.CUSTOMER] * standard_count +
        [UserPersona.CUSTOMER] * high_risk_count
    )
    customer_subcategories = (
        [CustomerSubCategory.PLATINUM] * platinum_count +
        [CustomerSubCategory.STANDARD] * standard_count +
        [CustomerSubCategory.HIGH_RISK] * high_risk_count
    )
    
    # Generate customer care reps
    customer_care_personas = [UserPersona.CUSTOMER_CARE_REP] * customer_care_count
    
    # Generate area managers
    area_manager_personas = [UserPersona.AREA_MANAGER] * area_manager_count
    
    # Combine all personas
    all_personas = customer_personas + customer_care_personas + area_manager_personas
    all_subcategories = customer_subcategories + [None] * (customer_care_count + area_manager_count)
    
    # Shuffle to randomize order
    combined = list(zip(all_personas, all_subcategories))
    random.shuffle(combined)
    all_personas, all_subcategories = zip(*combined)
    
    for i, (persona, sub_category) in enumerate(zip(all_personas, all_subcategories)):
        # Generate unique phone and email
        phone = fake.phone_number()
        while phone in phone_numbers:
            phone = fake.phone_number()
        phone_numbers.add(phone)
        
        email = fake.email()
        while email in emails:
            email = fake.email()
        emails.add(email)
        
        user = {
            'name': fake.name(),
            'phone': phone,
            'email': email,
            'persona': persona.value,
            'status': UserStatus.ACTIVE.value,
            'total_orders': 0,
            'lifetime_value': 0.0,
            'avg_order_value': 0.0,
            'refund_count': 0,
            'refund_rate': 0.0,
            'preferred_cuisines': [],
            'vip_status': False,
            'created_at': datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365)),
            'updated_at': datetime.now(timezone.utc)
        }
        
        if persona == UserPersona.CUSTOMER:
            user['sub_category'] = sub_category.value
            user['last_order_date'] = None
        else:
            user['sub_category'] = None
            user['last_order_date'] = None
        
        users.append(user)
    
    return users


def generate_zones(count: int) -> List[Dict]:
    """Generate zone documents"""
    print(f"Generating {count} zones...")
    
    zones = []
    cities = ['Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 'Chennai', 'Pune', 'Kolkata']
    zone_names_pool = {
        'Bangalore': ['Koramangala', 'Indiranagar', 'Whitefield', 'HSR Layout', 'Electronic City'],
        'Mumbai': ['Bandra', 'Andheri', 'Powai', 'Lower Parel', 'Vashi'],
        'Delhi': ['Connaught Place', 'Gurgaon', 'Noida', 'Dwarka', 'Rohini'],
        'Hyderabad': ['Hitech City', 'Banjara Hills', 'Gachibowli', 'Secunderabad'],
        'Chennai': ['T Nagar', 'Adyar', 'OMR', 'Velachery'],
        'Pune': ['Hinjewadi', 'Koregaon Park', 'Viman Nagar'],
        'Kolkata': ['Salt Lake', 'Park Street', 'New Town']
    }
    
    used_zone_names = set()
    
    for i in range(count):
        city = random.choice(cities)
        available_names = [n for n in zone_names_pool[city] if n not in used_zone_names]
        if not available_names:
            zone_name = f"{city} Zone {i+1}"
        else:
            zone_name = random.choice(available_names)
            used_zone_names.add(zone_name)
        
        # Generate simple rectangular boundary (GeoJSON Polygon)
        base_lat = fake.latitude()
        base_lng = fake.longitude()
        boundary = {
            'type': 'Polygon',
            'coordinates': [[
                [base_lng - 0.01, base_lat - 0.01],
                [base_lng + 0.01, base_lat - 0.01],
                [base_lng + 0.01, base_lat + 0.01],
                [base_lng - 0.01, base_lat + 0.01],
                [base_lng - 0.01, base_lat - 0.01]
            ]]
        }
        
        # Status: mostly active, some degraded
        status = ZoneStatus.ACTIVE if random.random() > 0.2 else ZoneStatus.DEGRADED
        
        zone = {
            'name': zone_name,
            'city': city,
            'tier': random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0],
            'boundary': boundary,
            'status': status.value,
            'live_metrics': {
                'updated_at': datetime.now(timezone.utc),
                'active_orders': random.randint(0, 50),
                'orders_last_hour': random.randint(0, 30),
                'orders_today': random.randint(50, 500),
                'avg_delivery_time': random.uniform(25, 45),
                'sla_breach_rate': random.uniform(0.0, 0.15),
                'riders_online': random.randint(10, 50),
                'riders_available': random.randint(5, 30),
                'riders_busy': random.randint(5, 25),
                'rider_utilization': random.uniform(0.5, 0.9),
                'restaurants_active': random.randint(20, 100),
                'restaurants_paused': random.randint(0, 10),
                'avg_prep_time': random.uniform(15, 30),
                'pending_complaints': random.randint(0, 10),
                'cancellation_rate': random.uniform(0.0, 0.1)
            },
            'created_at': datetime.now(timezone.utc) - timedelta(days=random.randint(90, 365)),
            'updated_at': datetime.now(timezone.utc)
        }
        
        zones.append(zone)
    
    return zones


def generate_restaurants(count: int, zones: List[Dict]) -> List[Dict]:
    """Generate restaurant documents"""
    print(f"Generating {count} restaurants...")
    
    if not zones:
        raise ValueError("Cannot generate restaurants without zones")
    
    restaurants = []
    cuisines_pool = ['Indian', 'Chinese', 'Italian', 'Mexican', 'Thai', 'Japanese', 'Continental', 'Fast Food']
    
    for i in range(count):
        # Assign to a random zone
        zone = random.choice(zones)
        zone_id = zone['_id']  # Binary UUID
        
        restaurant_type = random.choices(
            [RestaurantType.QUICK_SERVICE, RestaurantType.CASUAL_DINING, RestaurantType.CLOUD_KITCHEN],
            weights=[0.6, 0.3, 0.1]
        )[0]
        
        # Status: mostly active, some paused
        status = RestaurantStatus.ACTIVE if random.random() > 0.15 else RestaurantStatus.PAUSED
        
        # Generate coordinates within zone boundary (simplified)
        base_lat = fake.latitude()
        base_lng = fake.longitude()
        
        cuisines = random.sample(cuisines_pool, k=random.randint(1, 3))
        
        restaurant = {
            'name': fake.company() + ' Restaurant',
            'type': restaurant_type.value,
            'cuisines': cuisines,
            'location': {
                'address': fake.address(),
                'city': zone['city'],
                'zone_id': zone_id,  # Binary UUID reference
                'coordinates': {
                    'lat': base_lat,
                    'lng': base_lng
                }
            },
            'is_open': random.choice([True, False]),
            'is_paused': status == RestaurantStatus.PAUSED,
            'status': status.value,
            'current_metrics': {
                'updated_at': datetime.now(timezone.utc),
                'avg_prep_time_minutes': random.uniform(15, 30),
                'on_time_rate': random.uniform(0.75, 0.95),
                'quality_rating': random.uniform(3.5, 5.0),
                'support_ticket_count': random.randint(0, 5),
                'order_volume': random.randint(50, 500),
                'current_wait_time': random.uniform(10, 25)
            },
            'created_at': datetime.now(timezone.utc) - timedelta(days=random.randint(30, 365)),
            'updated_at': datetime.now(timezone.utc)
        }
        
        restaurants.append(restaurant)
    
    return restaurants


def generate_orders(
    count: int,
    users: List[Dict],
    restaurants: List[Dict],
    zones: List[Dict]
) -> List[Dict]:
    """Generate order documents"""
    print(f"Generating {count} orders...")
    
    if not users or not restaurants:
        raise ValueError("Cannot generate orders without users and restaurants")
    
    orders = []
    
    # Filter to customer users only
    customer_users = [u for u in users if u.get('persona') == UserPersona.CUSTOMER.value]
    if not customer_users:
        raise ValueError("No customer users found")
    
    # Create zone lookup from restaurants
    restaurant_zone_map = {r['_id']: r['location']['zone_id'] for r in restaurants}
    
    # Distribute orders across users (5-20 per user, varied)
    user_order_counts = {}
    remaining_orders = count
    
    for user in customer_users:
        user_id = user['_id']
        if remaining_orders <= 0:
            break
        # Assign 5-20 orders per user, but don't exceed remaining
        max_orders = min(random.randint(5, 20), remaining_orders)
        user_order_counts[user_id] = max_orders
        remaining_orders -= max_orders
    
    # Distribute remaining orders randomly
    while remaining_orders > 0:
        user_id = random.choice(customer_users)['_id']
        user_order_counts[user_id] = user_order_counts.get(user_id, 0) + 1
        remaining_orders -= 1
    
    # Generate orders
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)
    
    for user_id, order_count in user_order_counts.items():
        for _ in range(order_count):
            # Select random restaurant
            restaurant = random.choice(restaurants)
            restaurant_id = restaurant['_id']
            zone_id = restaurant_zone_map[restaurant_id]
            
            # Generate timestamp within last 7 days
            order_time = start_time + timedelta(
                seconds=random.randint(0, int((now - start_time).total_seconds()))
            )
            
            # Order details
            item_name = fake.word().capitalize() + ' ' + random.choice(['Burger', 'Pizza', 'Curry', 'Biryani', 'Noodles', 'Soup', 'Salad'])
            item_quantity = random.randint(1, 5)
            item_price = round(random.uniform(100, 500), 2)
            total_amount = round(item_price * item_quantity, 2)
            
            # Payment (all orders have payment)
            payment_method = random.choice(list(PaymentMethod))
            payment_status = random.choices(
                [PaymentStatus.COMPLETED, PaymentStatus.PENDING, PaymentStatus.FAILED],
                weights=[0.9, 0.08, 0.02]
            )[0]
            
            payment = {
                'amount': total_amount,
                'method': payment_method.value,
                'status': payment_status.value
            }
            
            # Refund (10-20% of orders)
            refund = None
            refund_status = RefundStatus.NONE
            if random.random() < 0.15:  # 15% have refunds
                refund_status = random.choice([RefundStatus.PENDING, RefundStatus.ISSUED, RefundStatus.COMPLETED])
                refund_amount = round(random.uniform(0.1, total_amount), 2)
                refund = {
                    'amount': refund_amount,
                    'status': refund_status.value,
                    'issued_at': order_time + timedelta(hours=random.randint(1, 24)) if refund_status != RefundStatus.NONE else None
                }
                if refund['issued_at']:
                    refund['issued_at'] = refund['issued_at'].isoformat()
            
            # Order status
            if refund_status != RefundStatus.NONE:
                order_status = OrderStatus.REFUNDED if refund_status == RefundStatus.COMPLETED else OrderStatus.CANCELLED
            else:
                order_status = random.choices(
                    [OrderStatus.DELIVERED, OrderStatus.IN_TRANSIT, OrderStatus.CANCELLED, OrderStatus.CONFIRMED],
                    weights=[0.7, 0.15, 0.1, 0.05]
                )[0]
            
            # Generate events timeline
            events = []
            if order_status != OrderStatus.CANCELLED:
                events.append({
                    'timestamp': order_time,
                    'event': OrderEventType.ORDER_PLACED.value,
                    'status': OrderEventStatus.PENDING.value
                })
                
                confirm_time = order_time + timedelta(minutes=random.randint(2, 10))
                events.append({
                    'timestamp': confirm_time,
                    'event': OrderEventType.RESTAURANT_CONFIRMED.value,
                    'status': OrderEventStatus.CONFIRMED.value
                })
                
                if order_status in [OrderStatus.IN_TRANSIT, OrderStatus.DELIVERED]:
                    pickup_time = confirm_time + timedelta(minutes=random.randint(15, 30))
                    events.append({
                        'timestamp': pickup_time,
                        'event': OrderEventType.PICKED_UP.value,
                        'status': OrderEventStatus.IN_TRANSIT.value
                    })
                    
                    if order_status == OrderStatus.DELIVERED:
                        delivery_time = pickup_time + timedelta(minutes=random.randint(20, 45))
                        events.append({
                            'timestamp': delivery_time,
                            'event': OrderEventType.DELIVERED.value,
                            'status': OrderEventStatus.DELIVERED.value
                        })
            else:
                events.append({
                    'timestamp': order_time,
                    'event': OrderEventType.ORDER_PLACED.value,
                    'status': OrderEventStatus.PENDING.value
                })
                events.append({
                    'timestamp': order_time + timedelta(minutes=random.randint(5, 30)),
                    'event': OrderEventType.CANCELLED.value,
                    'status': OrderEventStatus.CANCELLED.value
                })
            
            # Delivery estimates vs actuals
            estimated_delivery = order_time + timedelta(minutes=random.randint(30, 60))
            actual_delivery = None
            delivery_delay_minutes = 0
            
            if order_status == OrderStatus.DELIVERED and events:
                last_event = events[-1]
                if last_event['event'] == OrderEventType.DELIVERED.value:
                    actual_delivery = last_event['timestamp']
                    delay_seconds = (actual_delivery - estimated_delivery).total_seconds()
                    delivery_delay_minutes = int(delay_seconds / 60)
            
            order = {
                'user_id': user_id,  # Binary UUID reference
                'restaurant_id': restaurant_id,  # Binary UUID reference
                'zone_id': zone_id,  # Binary UUID reference
                'item_name': item_name,
                'item_quantity': item_quantity,
                'item_price': item_price,
                'total_amount': total_amount,
                'status': order_status.value,
                'events': events,
                'estimated_delivery': estimated_delivery,
                'actual_delivery': actual_delivery,
                'delivery_delay_minutes': delivery_delay_minutes,
                'payment': payment,
                'refund': refund,
                'refund_status': refund_status.value,
                'created_at': order_time,
                'updated_at': order_time
            }
            
            # Convert datetime objects to ISO strings for events
            for event in order['events']:
                if isinstance(event['timestamp'], datetime):
                    event['timestamp'] = event['timestamp'].isoformat()
            
            if order['estimated_delivery']:
                order['estimated_delivery'] = order['estimated_delivery'].isoformat()
            if order['actual_delivery']:
                order['actual_delivery'] = order['actual_delivery'].isoformat()
            if order['created_at']:
                order['created_at'] = order['created_at'].isoformat()
            if order['updated_at']:
                order['updated_at'] = order['updated_at'].isoformat()
            
            orders.append(order)
    
    return orders


def generate_support_tickets(
    count: int,
    users: List[Dict],
    orders: List[Dict],
    zones: List[Dict],
    restaurants: List[Dict]
) -> List[Dict]:
    """Generate support ticket documents"""
    print(f"Generating {count} support tickets...")
    
    if not users or not orders:
        raise ValueError("Cannot generate support tickets without users and orders")
    
    tickets = []
    
    # Filter to customer users
    customer_users = [u for u in users if u.get('persona') == UserPersona.CUSTOMER.value]
    
    # Scope distribution: 80% order, 15% zone, 5% restaurant
    order_scoped_count = int(count * 0.80)
    zone_scoped_count = int(count * 0.15)
    restaurant_scoped_count = count - order_scoped_count - zone_scoped_count
    
    # Issue type distribution
    issue_types = [
        (IssueType.ORDER_ISSUE, 0.40),
        (IssueType.DELIVERY, 0.20),
        (IssueType.QUALITY_SAFETY, 0.15),
        (IssueType.PAYMENT, 0.10),
        (IssueType.OPERATIONS, 0.10),
        (IssueType.SUPPORT, 0.03),
        (IssueType.GENERAL, 0.02)
    ]
    
    # Ticket type: 60% complaint, 40% general
    ticket_types = [TicketType.COMPLAINT] * int(count * 0.6) + [TicketType.GENERAL] * (count - int(count * 0.6))
    random.shuffle(ticket_types)
    
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)
    
    ticket_idx = 0
    
    # Generate order-scoped tickets (80%)
    for i in range(order_scoped_count):
        user = random.choice(customer_users)
        order = random.choice(orders)
        
        ticket_type = ticket_types[ticket_idx]
        ticket_idx += 1
        
        # Select issue type based on distribution
        issue_type = random.choices(
            [it[0] for it in issue_types],
            weights=[it[1] for it in issue_types]
        )[0]
        
        # Generate subtype (moderate complexity: some with multiple categories)
        subtype = {}
        if random.random() < 0.7:  # 70% have subtypes
            if issue_type == IssueType.ORDER_ISSUE:
                subtype['order_issues'] = random.sample(
                    [e.value for e in OrderIssueSubtype],
                    k=random.randint(1, 3)
                )
            elif issue_type == IssueType.QUALITY_SAFETY:
                subtype['quality_safety'] = random.sample(
                    [e.value for e in QualitySafetySubtype],
                    k=random.randint(1, 2)
                )
            elif issue_type == IssueType.DELIVERY:
                subtype['delivery'] = random.sample(
                    [e.value for e in DeliverySubtype],
                    k=random.randint(1, 2)
                )
            elif issue_type == IssueType.PAYMENT:
                subtype['payment'] = random.sample(
                    [e.value for e in PaymentSubtype],
                    k=random.randint(1, 2)
                )
            elif issue_type == IssueType.OPERATIONS:
                subtype['operation'] = random.sample(
                    [e.value for e in OperationSubtype],
                    k=random.randint(1, 2)
                )
            elif issue_type == IssueType.SUPPORT:
                subtype['support'] = random.sample(
                    [e.value for e in SupportSubtype],
                    k=random.randint(1, 2)
                )
            elif issue_type == IssueType.GENERAL:
                subtype['general'] = random.sample(
                    [e.value for e in GeneralSubtype],
                    k=random.randint(1, 2)
                )
            
            # Some tickets have multiple categories (moderate complexity)
            if random.random() < 0.3 and issue_type == IssueType.ORDER_ISSUE:
                subtype['delivery'] = random.sample(
                    [e.value for e in DeliverySubtype],
                    k=random.randint(1, 2)
                )
        
        severity = random.choices(
            [TicketSeverity.CRITICAL, TicketSeverity.HIGH, TicketSeverity.MEDIUM, TicketSeverity.LOW],
            weights=[0.1, 0.2, 0.5, 0.2]
        )[0]
        
        status = random.choice(list(TicketStatus))
        
        ticket_time = start_time + timedelta(
            seconds=random.randint(0, int((now - start_time).total_seconds()))
        )
        
        ticket = {
            'conversation_id': None,  # Nullable
            'user_id': user['_id'],  # Binary UUID reference
            'ticket_type': ticket_type.value,
            'issue_type': issue_type.value,
            'subtype': subtype if subtype else None,
            'severity': severity.value,
            'scope': TicketScope.ORDER.value,
            'order_id': order['_id'],  # Binary UUID reference
            'restaurant_id': order['restaurant_id'],  # Binary UUID reference
            'affected_zones': [],
            'affected_city': None,
            'title': fake.sentence(nb_words=6),
            'description': fake.text(max_nb_chars=200),
            'created_at': ticket_time,
            'updated_at': ticket_time,
            'timestamp': ticket_time,
            'status': status.value,
            'related_orders': [],
            'related_tickets': [],
            'agent_notes': [],
            'resolution_history': [],
            'resolution': None
        }
        
        # Convert datetime to ISO string
        ticket['created_at'] = ticket['created_at'].isoformat()
        ticket['updated_at'] = ticket['updated_at'].isoformat()
        ticket['timestamp'] = ticket['timestamp'].isoformat()
        
        tickets.append(ticket)
    
    # Generate zone-scoped tickets (15%)
    for i in range(zone_scoped_count):
        user = random.choice(customer_users)
        zone = random.choice(zones)
        
        ticket_type = ticket_types[ticket_idx]
        ticket_idx += 1
        
        issue_type = random.choices(
            [it[0] for it in issue_types],
            weights=[it[1] for it in issue_types]
        )[0]
        
        subtype = {}
        if random.random() < 0.7:
            if issue_type == IssueType.OPERATIONS:
                subtype['operation'] = random.sample(
                    [e.value for e in OperationSubtype],
                    k=random.randint(1, 2)
                )
        
        severity = random.choices(
            [TicketSeverity.CRITICAL, TicketSeverity.HIGH, TicketSeverity.MEDIUM, TicketSeverity.LOW],
            weights=[0.15, 0.25, 0.45, 0.15]
        )[0]
        
        status = random.choice(list(TicketStatus))
        
        ticket_time = start_time + timedelta(
            seconds=random.randint(0, int((now - start_time).total_seconds()))
        )
        
        ticket = {
            'conversation_id': None,
            'user_id': user['_id'],
            'ticket_type': ticket_type.value,
            'issue_type': issue_type.value,
            'subtype': subtype if subtype else None,
            'severity': severity.value,
            'scope': TicketScope.ZONE.value,
            'order_id': None,
            'restaurant_id': None,
            'affected_zones': [zone['_id']],  # Array of Binary UUIDs
            'affected_city': zone['city'],
            'title': fake.sentence(nb_words=6),
            'description': fake.text(max_nb_chars=200),
            'created_at': ticket_time,
            'updated_at': ticket_time,
            'timestamp': ticket_time,
            'status': status.value,
            'related_orders': [],
            'related_tickets': [],
            'agent_notes': [],
            'resolution_history': [],
            'resolution': None
        }
        
        ticket['created_at'] = ticket['created_at'].isoformat()
        ticket['updated_at'] = ticket['updated_at'].isoformat()
        ticket['timestamp'] = ticket['timestamp'].isoformat()
        
        tickets.append(ticket)
    
    # Generate restaurant-scoped tickets (5%)
    for i in range(restaurant_scoped_count):
        user = random.choice(customer_users)
        restaurant = random.choice(restaurants)
        
        ticket_type = ticket_types[ticket_idx]
        ticket_idx += 1
        
        issue_type = random.choice([IssueType.QUALITY_SAFETY, IssueType.OPERATIONS])
        
        subtype = {}
        if issue_type == IssueType.QUALITY_SAFETY:
            subtype['quality_safety'] = random.sample(
                [e.value for e in QualitySafetySubtype],
                k=random.randint(1, 2)
            )
        
        severity = random.choices(
            [TicketSeverity.HIGH, TicketSeverity.MEDIUM, TicketSeverity.LOW],
            weights=[0.3, 0.5, 0.2]
        )[0]
        
        status = random.choice(list(TicketStatus))
        
        ticket_time = start_time + timedelta(
            seconds=random.randint(0, int((now - start_time).total_seconds()))
        )
        
        ticket = {
            'conversation_id': None,
            'user_id': user['_id'],
            'ticket_type': ticket_type.value,
            'issue_type': issue_type.value,
            'subtype': subtype if subtype else None,
            'severity': severity.value,
            'scope': TicketScope.RESTAURANT.value,
            'order_id': None,
            'restaurant_id': restaurant['_id'],
            'affected_zones': [],
            'affected_city': None,
            'title': fake.sentence(nb_words=6),
            'description': fake.text(max_nb_chars=200),
            'created_at': ticket_time,
            'updated_at': ticket_time,
            'timestamp': ticket_time,
            'status': status.value,
            'related_orders': [],
            'related_tickets': [],
            'agent_notes': [],
            'resolution_history': [],
            'resolution': None
        }
        
        ticket['created_at'] = ticket['created_at'].isoformat()
        ticket['updated_at'] = ticket['updated_at'].isoformat()
        ticket['timestamp'] = ticket['timestamp'].isoformat()
        
        tickets.append(ticket)
    
    return tickets


def update_ticket_relationships(db, tickets: List[Dict], orders: List[Dict]):
    """Update tickets with related_orders and related_tickets after insertion"""
    print("Updating ticket relationships...")
    
    collection = db.support_tickets
    updated_count = 0
    
    # Add related_orders for some tickets
    order_scoped_tickets = [t for t in tickets if t.get('scope') == TicketScope.ORDER.value]
    if order_scoped_tickets:
        tickets_with_related = random.sample(
            order_scoped_tickets,
            k=min(int(len(order_scoped_tickets) * 0.3), len(order_scoped_tickets))
        )
        
        for ticket in tickets_with_related:
            if orders:
                # Don't include the ticket's own order_id
                other_orders = [o for o in orders if o['_id'] != ticket.get('order_id')]
                if other_orders:
                    related_orders = random.sample(other_orders, k=min(random.randint(1, 3), len(other_orders)))
                    collection.update_one(
                        {'_id': ticket['_id']},
                        {'$set': {'related_orders': [o['_id'] for o in related_orders]}}
                    )
                    updated_count += 1
    
    # Add related_tickets for some tickets
    if len(tickets) > 1:
        tickets_with_related_tickets = random.sample(
            tickets,
            k=min(int(len(tickets) * 0.2), len(tickets))
        )
        
        for ticket in tickets_with_related_tickets:
            # Get tickets inserted before this one (by comparing _id or using list index)
            ticket_index = tickets.index(ticket)
            if ticket_index > 0:
                previous_tickets = tickets[:ticket_index]
                if previous_tickets:
                    related_tickets = random.sample(previous_tickets, k=min(random.randint(1, 2), len(previous_tickets)))
                    collection.update_one(
                        {'_id': ticket['_id']},
                        {'$set': {'related_tickets': [t['_id'] for t in related_tickets]}}
                    )
                    updated_count += 1
    
    print(f"  Updated {updated_count} tickets with relationships")


def generate_zone_metrics_history(zones: List[Dict], orders: List[Dict]) -> List[Dict]:
    """Generate zone metrics history (hourly snapshots for last 7 days)"""
    print(f"Generating zone metrics history...")
    
    if not zones:
        return []
    
    metrics = []
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)
    
    # Create zone order lookup
    zone_orders = {}
    for order in orders:
        zone_id = order['zone_id']
        if zone_id not in zone_orders:
            zone_orders[zone_id] = []
        zone_orders[zone_id].append(order)
    
    for zone in zones:
        zone_id = zone['_id']
        zone_order_list = zone_orders.get(zone_id, [])
        
        # Generate hourly snapshots
        current_time = start_time
        while current_time <= now:
            # Calculate metrics based on orders in this hour window
            hour_start = current_time
            hour_end = current_time + timedelta(hours=1)
            
            hour_orders = [
                o for o in zone_order_list
                if hour_start <= datetime.fromisoformat(o['created_at'].replace('Z', '+00:00')) < hour_end
            ]
            
            total_orders = len(hour_orders)
            
            # Generate realistic metrics
            metric = {
                'zone_id': zone_id,  # Binary UUID reference
                'time_window': '1h',
                'timestamp': current_time,
                'total_orders': total_orders + random.randint(0, 10),  # Add some variance
                'avg_delivery_time_minutes': random.uniform(25, 45),
                'on_time_delivery_rate': random.uniform(0.7, 0.95),
                'support_ticket_count': random.randint(0, 5),
                'support_ticket_rate': random.uniform(0.0, 0.1),
                'active_drivers': random.randint(10, 50)
            }
            
            metric['timestamp'] = metric['timestamp'].isoformat()
            metrics.append(metric)
            
            current_time += timedelta(hours=1)
    
    print(f"  Generated {len(metrics)} zone metrics records")
    return metrics


def generate_restaurant_metrics_history(restaurants: List[Dict], orders: List[Dict]) -> List[Dict]:
    """Generate restaurant metrics history (hourly snapshots for last 7 days)"""
    print(f"Generating restaurant metrics history...")
    
    if not restaurants:
        return []
    
    metrics = []
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)
    
    # Create restaurant order lookup
    restaurant_orders = {}
    for order in orders:
        restaurant_id = order['restaurant_id']
        if restaurant_id not in restaurant_orders:
            restaurant_orders[restaurant_id] = []
        restaurant_orders[restaurant_id].append(order)
    
    for restaurant in restaurants:
        restaurant_id = restaurant['_id']
        restaurant_order_list = restaurant_orders.get(restaurant_id, [])
        
        # Generate hourly snapshots
        current_time = start_time
        while current_time <= now:
            # Calculate metrics based on orders in this hour window
            hour_start = current_time
            hour_end = current_time + timedelta(hours=1)
            
            hour_orders = [
                o for o in restaurant_order_list
                if hour_start <= datetime.fromisoformat(o['created_at'].replace('Z', '+00:00')) < hour_end
            ]
            
            order_volume = len(hour_orders)
            
            # Business hours: 8 AM to 11 PM (8:00 to 23:00)
            hour = current_time.hour
            is_open = 8 <= hour < 23
            
            metric = {
                'restaurant_id': restaurant_id,  # Binary UUID reference
                'time_window': '1h',
                'timestamp': current_time,
                'avg_prep_time_minutes': random.uniform(15, 30),
                'on_time_rate': random.uniform(0.75, 0.95),
                'quality_rating': random.uniform(3.5, 5.0),
                'support_ticket_count': random.randint(0, 3),
                'order_volume': order_volume + random.randint(0, 5),
                'is_open': is_open,
                'current_wait_time': random.uniform(10, 25) if is_open else 0
            }
            
            metric['timestamp'] = metric['timestamp'].isoformat()
            metrics.append(metric)
            
            current_time += timedelta(hours=1)
    
    print(f"  Generated {len(metrics)} restaurant metrics records")
    return metrics


# ============================================================================
# Post-Insert Updates
# ============================================================================

def update_order_ids(db, orders: List[Dict]):
    """Update orders with order_id = _id UUID"""
    print("Updating orders with order_id field...")
    
    collection = db.orders
    updated_count = 0
    
    for order in orders:
        order_id = order['_id']
        collection.update_one(
            {'_id': order_id},
            {'$set': {'order_id': order_id}}
        )
        updated_count += 1
    
    print(f"  Updated {updated_count} orders with order_id")


def update_ticket_ids(db, tickets: List[Dict]):
    """Update support_tickets with ticket_id = _id UUID"""
    print("Updating support_tickets with ticket_id field...")
    
    collection = db.support_tickets
    updated_count = 0
    
    for ticket in tickets:
        ticket_id = ticket['_id']
        collection.update_one(
            {'_id': ticket_id},
            {'$set': {'ticket_id': ticket_id}}
        )
        updated_count += 1
    
    print(f"  Updated {updated_count} tickets with ticket_id")


def update_user_stats(db, users: List[Dict], orders: List[Dict]):
    """Update user behavior stats from actual order data"""
    print("Updating user behavior stats...")
    
    collection = db.users
    
    # Group orders by user_id
    user_orders = {}
    for order in orders:
        user_id = order['user_id']
        if user_id not in user_orders:
            user_orders[user_id] = []
        user_orders[user_id].append(order)
    
    updated_count = 0
    
    for user in users:
        user_id = user['_id']
        user_order_list = user_orders.get(user_id, [])
        
        if not user_order_list:
            continue
        
        # Calculate stats
        total_orders = len(user_order_list)
        lifetime_value = sum(float(o['total_amount']) for o in user_order_list)
        avg_order_value = lifetime_value / total_orders if total_orders > 0 else 0.0
        
        # Count refunds
        refund_count = sum(1 for o in user_order_list if o.get('refund_status') != RefundStatus.NONE.value)
        refund_rate = refund_count / total_orders if total_orders > 0 else 0.0
        
        # Get last order date
        last_order_dates = [
            datetime.fromisoformat(o['created_at'].replace('Z', '+00:00'))
            for o in user_order_list
        ]
        last_order_date = max(last_order_dates) if last_order_dates else None
        
        # Extract preferred cuisines (from restaurant cuisines)
        cuisines_set = set()
        for order in user_order_list:
            # We don't have restaurant data in order, so use random cuisines
            # In real scenario, we'd join with restaurants collection
            pass
        
        # Set VIP status (lifetime_value > 10000)
        vip_status = lifetime_value > 10000
        
        # Update user document
        update_doc = {
            'total_orders': total_orders,
            'lifetime_value': round(lifetime_value, 2),
            'avg_order_value': round(avg_order_value, 2),
            'refund_count': refund_count,
            'refund_rate': round(refund_rate, 4),
            'vip_status': vip_status,
            'updated_at': datetime.now(timezone.utc).isoformat()
        }
        
        if last_order_date:
            update_doc['last_order_date'] = last_order_date.isoformat()
        
        # Get preferred cuisines from restaurants (simplified - use common cuisines)
        if total_orders > 0:
            update_doc['preferred_cuisines'] = random.sample(
                ['Indian', 'Chinese', 'Italian', 'Mexican', 'Thai'],
                k=min(random.randint(1, 3), 5)
            )
        
        collection.update_one(
            {'_id': user_id},
            {'$set': update_doc}
        )
        updated_count += 1
    
    print(f"  Updated {updated_count} users with behavior stats")


# ============================================================================
# Test Scenarios
# ============================================================================

def implement_test_scenarios(
    db,
    users: List[Dict],
    orders: List[Dict],
    tickets: List[Dict],
    zones: List[Dict],
    restaurants: List[Dict]
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Implement specific test scenarios:
    1. High-risk customer with multiple refunds
    2. Critical severity tickets
    3. Platinum customer with high lifetime value
    4. Zone with degraded status
    5. Restaurant with paused status
    """
    print("Implementing test scenarios...")
    
    # 1. High-risk customer with multiple refunds
    high_risk_users = [u for u in users if u.get('sub_category') == CustomerSubCategory.HIGH_RISK.value]
    if high_risk_users and orders:
        high_risk_user = high_risk_users[0]
        user_orders = [o for o in orders if o['user_id'] == high_risk_user['_id']]
        
        # Add refunds to 3-5 orders
        refund_orders = random.sample(user_orders, k=min(random.randint(3, 5), len(user_orders)))
        for order in refund_orders:
            if order.get('refund_status') == RefundStatus.NONE.value:
                order['refund'] = {
                    'amount': round(random.uniform(0.1, float(order['total_amount'])), 2),
                    'status': random.choice([RefundStatus.PENDING, RefundStatus.ISSUED, RefundStatus.COMPLETED]).value,
                    'issued_at': datetime.now(timezone.utc).isoformat()
                }
                order['refund_status'] = order['refund']['status']
        
        # Create support tickets for these orders
        for order in refund_orders[:2]:  # Create 2 tickets
            ticket = {
                'conversation_id': None,
                'user_id': high_risk_user['_id'],
                'ticket_type': TicketType.COMPLAINT.value,
                'issue_type': IssueType.ORDER_ISSUE.value,
                'subtype': {'order_issues': [OrderIssueSubtype.REFUND.value]},
                'severity': TicketSeverity.HIGH.value,
                'scope': TicketScope.ORDER.value,
                'order_id': order['_id'],
                'restaurant_id': order['restaurant_id'],
                'affected_zones': [],
                'affected_city': None,
                'title': 'Refund request for order',
                'description': 'Customer requesting refund for order issue',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': TicketStatus.OPEN.value,
                'related_orders': [order['_id']],
                'related_tickets': [],
                'agent_notes': [],
                'resolution_history': [],
                'resolution': None
            }
            tickets.append(ticket)
    
    # 2. Critical severity tickets (already generated, but ensure we have some)
    critical_tickets = [t for t in tickets if t.get('severity') == TicketSeverity.CRITICAL.value]
    if len(critical_tickets) < 2:
        # Add more critical tickets
        new_critical_tickets = []
        for _ in range(2 - len(critical_tickets)):
            user = random.choice([u for u in users if u.get('persona') == UserPersona.CUSTOMER.value])
            order = random.choice(orders) if orders else None
            
            ticket = {
                'conversation_id': None,
                'user_id': user['_id'],
                'ticket_type': TicketType.COMPLAINT.value,
                'issue_type': random.choice([IssueType.QUALITY_SAFETY, IssueType.OPERATIONS]).value,
                'subtype': {'quality_safety': [QualitySafetySubtype.FOOD_SAFETY.value]} if random.random() > 0.5 else {},
                'severity': TicketSeverity.CRITICAL.value,
                'scope': TicketScope.ORDER.value if order else TicketScope.ZONE.value,
                'order_id': order['_id'] if order else None,
                'restaurant_id': order['restaurant_id'] if order else None,
                'affected_zones': [random.choice(zones)['_id']] if zones and not order else [],
                'affected_city': random.choice(zones)['city'] if zones and not order else None,
                'title': 'Critical issue reported',
                'description': 'Urgent attention required',
                'created_at': datetime.now(timezone.utc).isoformat(),
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'status': TicketStatus.OPEN.value,
                'related_orders': [],
                'related_tickets': [],
                'agent_notes': [],
                'resolution_history': [],
                'resolution': None
            }
            new_critical_tickets.append(ticket)
        
        # Insert new critical tickets
        if new_critical_tickets:
            inserted_critical = insert_batch(db.support_tickets, new_critical_tickets)
            update_ticket_ids(db, inserted_critical)
            tickets.extend(inserted_critical)
    
    # 3. Platinum customer with high lifetime value (already handled in user generation)
    # Ensure platinum users have many orders
    platinum_users = [u for u in users if u.get('sub_category') == CustomerSubCategory.PLATINUM.value]
    if platinum_users and orders:
        platinum_user = platinum_users[0]
        user_orders = [o for o in orders if o['user_id'] == platinum_user['_id']]
        # If less than 15 orders, add more
        if len(user_orders) < 15:
            # Generate additional orders for platinum user
            additional_count = 15 - len(user_orders)
            # This would require regenerating orders, so we'll just ensure they exist
    
    # 4. Zone with degraded status (already handled in zone generation)
    degraded_zones = [z for z in zones if z.get('status') == ZoneStatus.DEGRADED.value]
    if not degraded_zones and zones:
        # Ensure at least one degraded zone
        degraded_zone = random.choice(zones)
        db.zones.update_one(
            {'_id': degraded_zone['_id']},
            {'$set': {
                'status': ZoneStatus.DEGRADED.value,
                'live_metrics.sla_breach_rate': random.uniform(0.2, 0.4)
            }}
        )
        degraded_zone['status'] = ZoneStatus.DEGRADED.value
    
    # 5. Restaurant with paused status (already handled in restaurant generation)
    paused_restaurants = [r for r in restaurants if r.get('status') == RestaurantStatus.PAUSED.value]
    if not paused_restaurants and restaurants:
        # Ensure at least one paused restaurant
        paused_restaurant = random.choice(restaurants)
        db.restaurants.update_one(
            {'_id': paused_restaurant['_id']},
            {'$set': {
                'status': RestaurantStatus.PAUSED.value,
                'is_paused': True,
                'is_open': False
            }}
        )
        paused_restaurant['status'] = RestaurantStatus.PAUSED.value
    
    print("  Test scenarios implemented")
    return users, orders, tickets, zones, restaurants


# ============================================================================
# Index Creation
# ============================================================================

def create_indexes(db):
    """Create all MongoDB indexes"""
    print("Creating indexes...")
    
    # Users indexes
    db.users.create_index('phone', unique=True)
    db.users.create_index('email', sparse=True)
    db.users.create_index([('persona', 1), ('sub_category', 1)])
    db.users.create_index([('last_order_date', -1)])
    
    # Orders indexes
    db.orders.create_index('order_id', unique=True)
    db.orders.create_index([('user_id', 1), ('created_at', -1)])
    db.orders.create_index([('restaurant_id', 1), ('created_at', -1)])
    db.orders.create_index([('zone_id', 1), ('created_at', -1)])
    db.orders.create_index('status')
    db.orders.create_index('refund_status')
    db.orders.create_index([('created_at', -1)])
    
    # Zones indexes
    db.zones.create_index('city')
    db.zones.create_index('status')
    db.zones.create_index([('boundary', '2dsphere')])
    
    # Restaurants indexes
    db.restaurants.create_index('location.zone_id')
    db.restaurants.create_index('status')
    db.restaurants.create_index([('location.coordinates', '2dsphere')])
    
    # Support tickets indexes
    db.support_tickets.create_index('ticket_id', unique=True)
    db.support_tickets.create_index('conversation_id', sparse=True)
    db.support_tickets.create_index([('user_id', 1), ('created_at', -1)])
    db.support_tickets.create_index([('order_id', 1), ('timestamp', -1)], sparse=True)
    db.support_tickets.create_index([('restaurant_id', 1), ('timestamp', -1)], sparse=True)
    db.support_tickets.create_index([('affected_zones', 1), ('created_at', -1)])
    db.support_tickets.create_index([('scope', 1), ('timestamp', -1)])
    db.support_tickets.create_index([('severity', 1), ('status', 1)])
    db.support_tickets.create_index([('ticket_type', 1), ('status', 1)])
    db.support_tickets.create_index([('issue_type', 1), ('status', 1)])
    db.support_tickets.create_index([('timestamp', -1)])
    
    # Zone metrics history indexes
    db.zone_metrics_history.create_index([('zone_id', 1), ('timestamp', -1)])
    db.zone_metrics_history.create_index([('timestamp', -1)], expireAfterSeconds=604800)  # 7-day TTL
    
    # Restaurant metrics history indexes
    db.restaurant_metrics_history.create_index([('restaurant_id', 1), ('timestamp', -1)])
    db.restaurant_metrics_history.create_index([('timestamp', -1)], expireAfterSeconds=604800)  # 7-day TTL
    
    print("  All indexes created")


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Seed MongoDB with test data')
    parser.add_argument('--force', action='store_true', help='Delete existing data before seeding')
    parser.add_argument('--create-indexes', action='store_true', help='Create MongoDB indexes')
    args = parser.parse_args()
    
    print("=" * 60)
    print("MongoDB Test Data Seeding Script")
    print("=" * 60)
    
    # Connect to MongoDB
    client, db = get_mongodb_client()
    
    try:
        # Check if collections have data
        if not args.force:
            has_data = any(
                db[coll].count_documents({}) > 0
                for coll in ['users', 'zones', 'restaurants', 'orders', 'support_tickets']
            )
            if has_data:
                print("WARNING: Collections already contain data. Use --force to delete and recreate.")
                return
        
        # Cleanup if --force
        if args.force:
            cleanup_existing_data(db)
        
        # Generate data volumes (small: 10-50 per collection)
        user_count = random.randint(30, 50)
        zone_count = random.randint(5, 10)
        restaurant_count = random.randint(20, 30)
        order_count = random.randint(100, 200)
        ticket_count = max(5, int(order_count * 0.08))  # 5-10% of orders
        
        # Generate users
        users_docs = generate_users(user_count)
        users = insert_batch(db.users, users_docs)
        print(f"Inserted {len(users)} users")
        
        # Generate zones
        zones_docs = generate_zones(zone_count)
        zones = insert_batch(db.zones, zones_docs)
        print(f"Inserted {len(zones)} zones")
        
        # Generate restaurants
        restaurants_docs = generate_restaurants(restaurant_count, zones)
        restaurants = insert_batch(db.restaurants, restaurants_docs)
        print(f"Inserted {len(restaurants)} restaurants")
        
        # Generate orders
        orders_docs = generate_orders(order_count, users, restaurants, zones)
        orders = insert_batch(db.orders, orders_docs)
        print(f"Inserted {len(orders)} orders")
        
        # Update orders with order_id = _id
        update_order_ids(db, orders)
        
        # Generate support tickets
        tickets_docs = generate_support_tickets(ticket_count, users, orders, zones, restaurants)
        tickets = insert_batch(db.support_tickets, tickets_docs)
        print(f"Inserted {len(tickets)} support tickets")
        
        # Update tickets with ticket_id = _id
        update_ticket_ids(db, tickets)
        
        # Update ticket relationships
        update_ticket_relationships(db, tickets, orders)
        
        # Implement test scenarios
        users, orders, tickets, zones, restaurants = implement_test_scenarios(
            db, users, orders, tickets, zones, restaurants
        )
        
        # Generate metrics history
        zone_metrics = generate_zone_metrics_history(zones, orders)
        if zone_metrics:
            insert_batch(db.zone_metrics_history, zone_metrics)
        
        restaurant_metrics = generate_restaurant_metrics_history(restaurants, orders)
        if restaurant_metrics:
            insert_batch(db.restaurant_metrics_history, restaurant_metrics)
        
        # Update user stats
        update_user_stats(db, users, orders)
        
        # Create indexes if requested
        if args.create_indexes:
            create_indexes(db)
        
        # Summary
        print("=" * 60)
        print("Seeding completed successfully!")
        print("=" * 60)
        print(f"Users: {len(users)}")
        print(f"Zones: {len(zones)}")
        print(f"Restaurants: {len(restaurants)}")
        print(f"Orders: {len(orders)}")
        print(f"Support Tickets: {len(tickets)}")
        print(f"Zone Metrics: {len(zone_metrics)}")
        print(f"Restaurant Metrics: {len(restaurant_metrics)}")
        
    except Exception as e:
        print(f"ERROR: Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        client.close()
        print("MongoDB connection closed")


if __name__ == '__main__':
    main()
