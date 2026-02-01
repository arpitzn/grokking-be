"""
MongoDB Collections Migration Script

This script migrates data from old collections to new streamlined collections:
1. incidents → support_tickets
2. cases → support_tickets
3. customers → users
4. orders: Update customer_id → user_id, add embedded refund/payment objects, remove delivery_partner_id

Run with: python scripts/migrate_mongodb_collections.py
"""

import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from app.infra.config import settings


async def migrate_incidents_to_support_tickets(db):
    """Migrate incidents collection to support_tickets"""
    print("Migrating incidents → support_tickets...")
    
    incidents = db.incidents
    support_tickets = db.support_tickets
    
    async for incident in incidents.find({}):
        # Map incident to support_ticket schema
        ticket = {
            "ticket_id": incident.get("incident_id", f"TICKET-{datetime.now().strftime('%Y%m%d')}-{incident.get('_id')}"),
            "conversation_id": incident.get("conversation_id", ""),  # May need to be set manually
            "user_id": incident.get("customer_id", ""),  # Map customer_id to user_id
            "ticket_type": "complaint" if incident.get("type") else "general",
            "issue_type": _map_incident_type_to_issue_type(incident.get("type", "")),
            "subtype": _map_incident_to_subtype(incident.get("type", ""), incident.get("category", "")),
            "severity": incident.get("severity", 3),
            "scope": incident.get("scope", "order"),
            "affected_zones": incident.get("affected_zones", []),
            "affected_city": incident.get("affected_city") if incident.get("scope") == "zone" else None,
            "order_id": incident.get("order_id"),
            "restaurant_id": incident.get("restaurant_id"),
            "title": incident.get("title", ""),
            "description": incident.get("description", ""),
            "created_at": incident.get("created_at", datetime.now()),
            "updated_at": incident.get("updated_at", datetime.now()),
            "timestamp": incident.get("timestamp", incident.get("created_at", datetime.now())),
            "status": _map_incident_status(incident.get("status", "detected")),
            "related_orders": [incident.get("order_id")] if incident.get("order_id") else [],
            "related_tickets": [],
            "agent_notes": [],
            "resolution_history": [],
            "resolution": incident.get("resolution")
        }
        
        await support_tickets.insert_one(ticket)
    
    count = await incidents.count_documents({})
    print(f"Migrated {count} incidents to support_tickets")


async def migrate_cases_to_support_tickets(db):
    """Migrate cases collection to support_tickets"""
    print("Migrating cases → support_tickets...")
    
    cases = db.cases
    support_tickets = db.support_tickets
    
    async for case in cases.find({}):
        # Check if ticket already exists (from incidents migration)
        existing = await support_tickets.find_one({"ticket_id": case.get("case_id")})
        if existing:
            # Update existing ticket with case data
            await support_tickets.update_one(
                {"ticket_id": case.get("case_id")},
                {
                    "$set": {
                        "related_orders": case.get("related_orders", []),
                        "related_tickets": case.get("related_cases", []),
                        "agent_notes": case.get("agent_notes", []),
                        "resolution_history": case.get("resolution_history", []),
                        "status": case.get("status", "open")
                    }
                }
            )
        else:
            # Create new ticket from case
            ticket = {
                "ticket_id": case.get("case_id", f"TICKET-{datetime.now().strftime('%Y%m%d')}-{case.get('_id')}"),
                "conversation_id": "",  # May need to be set manually
                "user_id": "",  # May need to be set manually
                "ticket_type": "general",
                "issue_type": "general",
                "subtype": {},
                "severity": _map_priority_to_severity(case.get("priority", "low")),
                "scope": "order",
                "affected_zones": [],
                "affected_city": None,
                "order_id": None,
                "restaurant_id": None,
                "title": "",
                "description": "",
                "created_at": case.get("created_at", datetime.now()),
                "updated_at": datetime.now(),
                "timestamp": case.get("created_at", datetime.now()),
                "status": case.get("status", "open"),
                "related_orders": case.get("related_orders", []),
                "related_tickets": case.get("related_cases", []),
                "agent_notes": case.get("agent_notes", []),
                "resolution_history": case.get("resolution_history", []),
                "resolution": None
            }
            
            await support_tickets.insert_one(ticket)
    
    count = await cases.count_documents({})
    print(f"Migrated {count} cases to support_tickets")


async def migrate_customers_to_users(db):
    """Migrate customers collection to users"""
    print("Migrating customers → users...")
    
    customers = db.customers
    users = db.users
    
    async for customer in customers.find({}):
        user = {
            "user_id": customer.get("customer_id", f"USER-{customer.get('_id')}"),
            "name": customer.get("name", ""),
            "phone": customer.get("phone", ""),
            "email": customer.get("email", ""),
            "created_at": customer.get("created_at", datetime.now()),
            "persona": "customer",  # Default persona
            "sub_category": _map_tier_to_sub_category(customer.get("tier", "standard")),
            "status": customer.get("status", "active"),
            "total_orders": customer.get("total_orders", 0),
            "lifetime_value": customer.get("lifetime_value", 0),
            "avg_order_value": customer.get("avg_order_value", 0),
            "refund_count": customer.get("refund_count", 0),
            "refund_rate": customer.get("refund_rate", 0),
            "last_order_date": customer.get("last_order_date"),
            "preferred_cuisines": customer.get("preferred_cuisines", []),
            "vip_status": customer.get("vip_status", False),
            "updated_at": customer.get("updated_at", datetime.now())
        }
        
        await users.insert_one(user)
    
    count = await customers.count_documents({})
    print(f"Migrated {count} customers to users")


async def migrate_orders(db):
    """Update orders collection: customer_id → user_id, add embedded objects, remove delivery_partner_id"""
    print("Updating orders collection...")
    
    orders = db.orders
    
    # Get refunds collection for reference (if exists)
    refunds = db.refunds
    payments = db.payments
    
    updated_count = 0
    
    async for order in orders.find({}):
        update_doc = {}
        
        # Map customer_id to user_id
        if order.get("customer_id"):
            # Find corresponding user_id
            user = await db.users.find_one({"user_id": order.get("customer_id")})
            if not user:
                # Try to find by old customer_id pattern
                user = await db.users.find_one({"user_id": order.get("customer_id")})
            if user:
                update_doc["user_id"] = user.get("user_id")
            else:
                # Keep customer_id as user_id if user not found (may need manual fix)
                update_doc["user_id"] = order.get("customer_id")
        
        # Add embedded refund if refunds collection exists
        if refunds:
            refund = await refunds.find_one({"order_id": order.get("order_id")})
            if refund:
                update_doc["refund"] = {
                    "amount": refund.get("amount", 0),
                    "status": refund.get("status", "none"),
                    "issued_at": refund.get("issued_at")
                }
                update_doc["refund_status"] = refund.get("status", "none")
        
        # Add embedded payment if payments collection exists
        if payments:
            payment = await payments.find_one({"order_id": order.get("order_id")})
            if payment:
                update_doc["payment"] = {
                    "amount": payment.get("amount", 0),
                    "method": payment.get("method", "cash"),
                    "status": payment.get("status", "pending")
                }
        
        # Remove delivery_partner_id
        update_doc["$unset"] = {"delivery_partner_id": "", "customer_id": ""}
        
        if update_doc:
            await orders.update_one(
                {"_id": order.get("_id")},
                update_doc
            )
            updated_count += 1
    
    print(f"Updated {updated_count} orders")


def _map_incident_type_to_issue_type(incident_type: str) -> str:
    """Map incident type to issue_type enum"""
    mapping = {
        "delivery_delay": "delivery",
        "rider_shortage": "operations",
        "restaurant_outage": "operations",
        "zone_delay": "operations",
        "quality_issue": "quality_safety",
        "safety_issue": "quality_safety",
        "payment_issue": "payment"
    }
    return mapping.get(incident_type.lower(), "operations")


def _map_incident_to_subtype(incident_type: str, category: str) -> dict:
    """Map incident to subtype object"""
    subtype = {}
    
    if "delivery" in incident_type.lower():
        subtype["delivery"] = ["delivery", "late_delivery"]
    elif "quality" in incident_type.lower() or category == "quality":
        subtype["quality_safety"] = ["quality"]
    elif "safety" in incident_type.lower() or category == "safety":
        subtype["quality_safety"] = ["food_safety"]
    elif "payment" in incident_type.lower():
        subtype["payment"] = ["payment"]
    elif category == "operations":
        subtype["operation"] = ["incident"]
    
    return subtype


def _map_incident_status(status: str) -> str:
    """Map incident status to ticket status"""
    mapping = {
        "detected": "open",
        "acknowledged": "pending",
        "in_progress": "pending",
        "resolved": "resolved",
        "closed": "closed"
    }
    return mapping.get(status.lower(), "open")


def _map_priority_to_severity(priority: str) -> int:
    """Map priority to severity (1=Critical, 2=High, 3=Medium, 4=Low)"""
    mapping = {
        "high": 2,
        "medium": 3,
        "low": 4
    }
    return mapping.get(priority.lower(), 3)


def _map_tier_to_sub_category(tier: str) -> str:
    """Map customer tier to sub_category"""
    mapping = {
        "vip": "platinum",
        "gold": "platinum",
        "regular": "standard",
        "new": "standard"
    }
    return mapping.get(tier.lower(), "standard")


async def main():
    """Main migration function"""
    print("Starting MongoDB collections migration...")
    print(f"Connecting to MongoDB: {settings.mongodb_uri}")
    
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db_name]
    
    try:
        # Run migrations
        await migrate_incidents_to_support_tickets(db)
        await migrate_cases_to_support_tickets(db)
        await migrate_customers_to_users(db)
        await migrate_orders(db)
        
        print("\nMigration completed successfully!")
        print("\nNext steps:")
        print("1. Verify migrated data")
        print("2. Update application code to use new collections")
        print("3. Drop old collections after verification:")
        print("   - db.incidents.drop()")
        print("   - db.cases.drop()")
        print("   - db.customers.drop()")
        print("   - db.delivery_partners.drop()")
        print("   - db.ratings.drop()")
        print("   - db.order_items.drop()")
        
    except Exception as e:
        print(f"Migration failed: {e}")
        raise
    finally:
        client.close()


if __name__ == "__main__":
    asyncio.run(main())
