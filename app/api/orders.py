"""Orders API endpoint"""
from fastapi import APIRouter, HTTPException
from app.infra.mongo import get_mongodb_client
from app.models.mongodb_schemas import (
    OrderStatus, PaymentMethod, PaymentStatus, RefundStatus,
    OrderEventType, OrderEventStatus
)
from app.utils.logging_utils import (
    log_request_start, log_request_end, log_db_operation, log_error_with_context
)
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import time
from datetime import datetime, timezone
from bson import Binary, ObjectId
from uuid import UUID
from bson.binary import UuidRepresentation
from app.utils.uuid_helpers import is_uuid_string, uuid_to_binary

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orders", tags=["orders"])


# Request/Response models
class CreateOrderRequest(BaseModel):
    """Request schema for creating an order"""
    user_id: str = Field(..., description="User ID")
    restaurant_id: str = Field(..., description="Restaurant ID")
    zone_id: str = Field(..., description="Zone ID")
    item_name: str = Field(..., description="Item name")
    item_quantity: int = Field(..., gt=0, description="Item quantity")
    item_price: float = Field(..., ge=0, description="Item price")
    payment_method: str = Field(..., description="Payment method")
    estimated_delivery: str = Field(..., description="Estimated delivery datetime (ISO format)")


class UpdateOrderRequest(BaseModel):
    """Request schema for updating an order"""
    status: Optional[str] = None
    payment: Optional[Dict[str, Any]] = None


class OrderItem(BaseModel):
    """Order item schema"""
    order_id: str
    user_id: str
    restaurant_id: str
    zone_id: str
    item_name: str
    item_quantity: int
    item_price: float
    total_amount: float
    status: str
    created_at: str
    updated_at: str
    payment: Dict[str, Any]
    refund: Optional[Dict[str, Any]] = None
    refund_status: Optional[str] = None
    events: List[Dict[str, Any]] = []
    estimated_delivery: Optional[str] = None
    actual_delivery: Optional[str] = None
    delivery_delay_minutes: Optional[int] = None


class OrdersResponse(BaseModel):
    """Response schema for orders list"""
    orders: List[OrderItem]
    count: int


def serialize_order(order: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize MongoDB order document to API response format"""
    # Convert Binary UUID to string for order_id
    order_id = order.get("_id")
    if order_id and isinstance(order_id, Binary):
        order_id = str(order_id.as_uuid())
    elif order_id and isinstance(order_id, ObjectId):
        order_id = str(order_id)
    elif order_id:
        order_id = str(order_id)
    
    # Also check for order_id field
    if "order_id" in order:
        order_id_field = order.get("order_id")
        if order_id_field and isinstance(order_id_field, Binary):
            order_id = str(order_id_field.as_uuid())
        elif order_id_field and isinstance(order_id_field, ObjectId):
            order_id = str(order_id_field)
        elif order_id_field:
            order_id = str(order_id_field)
    
    # Convert Binary UUIDs to strings
    user_id = order.get("user_id")
    if user_id and isinstance(user_id, Binary):
        user_id = str(user_id.as_uuid())
    elif user_id and isinstance(user_id, ObjectId):
        user_id = str(user_id)
    elif user_id:
        user_id = str(user_id)
    
    restaurant_id = order.get("restaurant_id")
    if restaurant_id and isinstance(restaurant_id, Binary):
        restaurant_id = str(restaurant_id.as_uuid())
    elif restaurant_id and isinstance(restaurant_id, ObjectId):
        restaurant_id = str(restaurant_id)
    elif restaurant_id:
        restaurant_id = str(restaurant_id)
    
    zone_id = order.get("zone_id")
    if zone_id and isinstance(zone_id, Binary):
        zone_id = str(zone_id.as_uuid())
    elif zone_id and isinstance(zone_id, ObjectId):
        zone_id = str(zone_id)
    elif zone_id:
        zone_id = str(zone_id)
    
    # Serialize datetime fields
    created_at = order.get("created_at")
    if created_at:
        if isinstance(created_at, str):
            created_at = created_at
        else:
            created_at = created_at.isoformat() if hasattr(created_at, 'isoformat') else str(created_at)
    
    updated_at = order.get("updated_at")
    if updated_at:
        if isinstance(updated_at, str):
            updated_at = updated_at
        else:
            updated_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
    
    estimated_delivery = order.get("estimated_delivery")
    if estimated_delivery:
        if isinstance(estimated_delivery, str):
            estimated_delivery = estimated_delivery
        else:
            estimated_delivery = estimated_delivery.isoformat() if hasattr(estimated_delivery, 'isoformat') else str(estimated_delivery)
    
    actual_delivery = order.get("actual_delivery")
    if actual_delivery:
        if isinstance(actual_delivery, str):
            actual_delivery = actual_delivery
        else:
            actual_delivery = actual_delivery.isoformat() if hasattr(actual_delivery, 'isoformat') else str(actual_delivery)
    
    # Serialize events
    events = order.get("events", [])
    serialized_events = []
    for event in events:
        event_copy = event.copy()
        timestamp = event.get("timestamp")
        if timestamp:
            if isinstance(timestamp, str):
                event_copy["timestamp"] = timestamp
            else:
                event_copy["timestamp"] = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp)
        serialized_events.append(event_copy)
    
    return {
        "order_id": order_id,
        "user_id": user_id,
        "restaurant_id": restaurant_id,
        "zone_id": zone_id,
        "item_name": order.get("item_name", ""),
        "item_quantity": order.get("item_quantity", 0),
        "item_price": order.get("item_price", 0.0),
        "total_amount": order.get("total_amount", 0.0),
        "status": order.get("status", ""),
        "created_at": created_at,
        "updated_at": updated_at,
        "payment": order.get("payment", {}),
        "refund": order.get("refund"),
        "refund_status": order.get("refund_status"),
        "events": serialized_events,
        "estimated_delivery": estimated_delivery,
        "actual_delivery": actual_delivery,
        "delivery_delay_minutes": order.get("delivery_delay_minutes"),
    }


def uuid_string_to_binary(uuid_string: str) -> Binary:
    """Convert UUID string to BSON Binary UUID
    
    Handles both standard UUID format and ObjectId-like strings.
    If the string is not a valid UUID, tries to convert it as-is (for ObjectId compatibility).
    """
    try:
        # Try to parse as UUID first
        uuid_obj = UUID(uuid_string)
        return Binary.from_uuid(uuid_obj, uuid_representation=UuidRepresentation.STANDARD)
    except (ValueError, TypeError):
        # If not a valid UUID, check if it's an ObjectId format (24 hex chars)
        # or try to use it directly - MongoDB might accept it
        if len(uuid_string) == 24 and all(c in '0123456789abcdef' for c in uuid_string.lower()):
            # This looks like an ObjectId - we need to handle this differently
            # For now, raise an error to indicate the issue
            raise ValueError(f"Invalid UUID format (appears to be ObjectId): {uuid_string}. Expected UUID format.")
        raise ValueError(f"Invalid UUID format: {uuid_string}")


@router.get("/{user_id}", response_model=OrdersResponse)
async def get_orders(user_id: str):
    """Get orders for a user (End Customer only)"""
    start_time = time.time()
    log_request_start(logger, "GET", f"/api/orders/{user_id}", user_id=user_id)
    
    try:
        db = await get_mongodb_client()
        
        # Lookup user and get their _id (handles Binary UUID, ObjectId, or string)
        user_doc = None
        
        # Approach 1: Try to find user by user_id field first
        user_doc = await db.users.find_one({"user_id": user_id})
        
        # Approach 2: Try as UUID string and convert to Binary UUID
        if not user_doc and is_uuid_string(user_id):
            try:
                user_id_binary_query = uuid_to_binary(user_id)
                user_doc = await db.users.find_one({"_id": user_id_binary_query})
            except (ValueError, TypeError):
                pass
        
        # Approach 3: If not found and looks like ObjectId (24 hex chars), try as ObjectId
        if not user_doc and len(user_id) == 24:
            try:
                if all(c in '0123456789abcdef' for c in user_id.lower()):
                    user_id_objid = ObjectId(user_id)
                    user_doc = await db.users.find_one({"_id": user_id_objid})
            except (ValueError, TypeError):
                pass
        
        # Get the _id from user document (could be Binary UUID or ObjectId)
        if user_doc and user_doc.get("_id"):
            user_id_for_query = user_doc["_id"]
        else:
            # Fallback: Try querying orders directly with the string (in case orders.user_id is stored as string)
            query = {"user_id": user_id}
            cursor = db.orders.find(query).sort([("created_at", -1)])
            orders_raw = await cursor.to_list(length=None)
            orders = [serialize_order(order) for order in orders_raw]
            
            log_request_end(
                logger, "GET", f"/api/orders/{user_id}",
                status_code=200,
                duration_ms=(time.time() - start_time) * 1000,
                details={"order_count": len(orders), "note": "queried_with_string_user_id"},
                user_id=user_id
            )
            
            return {
                "orders": orders,
                "count": len(orders)
            }
        
        # Query orders for this user using the user's _id (Binary UUID or ObjectId)
        query = {"user_id": user_id_for_query}
        
        # Fetch orders sorted by created_at DESC (newest first)
        cursor = db.orders.find(query).sort([("created_at", -1)])
        orders_raw = await cursor.to_list(length=None)
        
        # Serialize orders
        orders = [serialize_order(order) for order in orders_raw]
        
        # Log DB result validation
        log_db_operation(
            logger, "find", "orders",
            result_count=len(orders),
            expected=False,  # Empty is valid for new users
            user_id=user_id
        )
        
        log_request_end(
            logger, "GET", f"/api/orders/{user_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"order_count": len(orders)},
            user_id=user_id
        )
        
        return {
            "orders": orders,
            "count": len(orders)
        }
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "get_orders_error",
            context={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=OrderItem)
async def create_order(request: CreateOrderRequest):
    """Create a new order"""
    start_time = time.time()
    log_request_start(logger, "POST", "/api/orders", body=request.model_dump())
    
    try:
        db = await get_mongodb_client()
        
        # Lookup user and get their _id (handles Binary UUID, ObjectId, or string)
        user_doc = None
        
        # Approach 1: Try to find user by user_id field
        user_doc = await db.users.find_one({"user_id": request.user_id})
        
        # Approach 2: Try as UUID string and convert to Binary UUID
        if not user_doc and is_uuid_string(request.user_id):
            try:
                user_id_binary_query = uuid_to_binary(request.user_id)
                user_doc = await db.users.find_one({"_id": user_id_binary_query})
            except (ValueError, TypeError):
                pass
        
        # Approach 3: If not found and looks like ObjectId (24 hex chars), try as ObjectId
        if not user_doc and len(request.user_id) == 24:
            try:
                if all(c in '0123456789abcdef' for c in request.user_id.lower()):
                    user_id_objid = ObjectId(request.user_id)
                    user_doc = await db.users.find_one({"_id": user_id_objid})
            except (ValueError, TypeError):
                pass
        
        if not user_doc or not user_doc.get("_id"):
            raise HTTPException(status_code=404, detail=f"User not found: {request.user_id}")
        
        user_id_binary = user_doc["_id"]
        
        # Lookup restaurant and get its _id (handles Binary UUID, ObjectId, or string)
        restaurant_doc = None
        
        # Approach 1: Try as UUID string and convert to Binary UUID
        if is_uuid_string(request.restaurant_id):
            try:
                restaurant_id_binary_query = uuid_to_binary(request.restaurant_id)
                restaurant_doc = await db.restaurants.find_one({"_id": restaurant_id_binary_query})
            except (ValueError, TypeError):
                pass
        
        # Approach 2: If not found and looks like ObjectId (24 hex chars), try as ObjectId
        if not restaurant_doc and len(request.restaurant_id) == 24:
            try:
                if all(c in '0123456789abcdef' for c in request.restaurant_id.lower()):
                    restaurant_id_objid = ObjectId(request.restaurant_id)
                    restaurant_doc = await db.restaurants.find_one({"_id": restaurant_id_objid})
            except (ValueError, TypeError):
                pass
        
        if not restaurant_doc or not restaurant_doc.get("_id"):
            raise HTTPException(status_code=404, detail=f"Restaurant not found: {request.restaurant_id}")
        
        restaurant_id_binary = restaurant_doc["_id"]
        
        # Lookup zone and get its _id (handles Binary UUID, ObjectId, or string)
        zone_doc = None
        
        # Approach 1: Try as UUID string and convert to Binary UUID
        if is_uuid_string(request.zone_id):
            try:
                zone_id_binary_query = uuid_to_binary(request.zone_id)
                zone_doc = await db.zones.find_one({"_id": zone_id_binary_query})
            except (ValueError, TypeError):
                pass
        
        # Approach 2: If not found and looks like ObjectId (24 hex chars), try as ObjectId
        if not zone_doc and len(request.zone_id) == 24:
            try:
                if all(c in '0123456789abcdef' for c in request.zone_id.lower()):
                    zone_id_objid = ObjectId(request.zone_id)
                    zone_doc = await db.zones.find_one({"_id": zone_id_objid})
            except (ValueError, TypeError):
                pass
        
        if not zone_doc or not zone_doc.get("_id"):
            raise HTTPException(status_code=404, detail=f"Zone not found: {request.zone_id}")
        
        zone_id_binary = zone_doc["_id"]
        
        # Calculate total amount
        total_amount = round(request.item_price * request.item_quantity, 2)
        
        # Create initial event
        now = datetime.now(timezone.utc)
        initial_event = {
            "timestamp": now.isoformat(),
            "event": OrderEventType.ORDER_PLACED.value,
            "status": OrderEventStatus.PENDING.value
        }
        
        # Create payment object
        payment = {
            "amount": total_amount,
            "method": request.payment_method,
            "status": PaymentStatus.PENDING.value
        }
        
        # Create order document
        order_doc = {
            "user_id": user_id_binary,
            "restaurant_id": restaurant_id_binary,
            "zone_id": zone_id_binary,
            "item_name": request.item_name,
            "item_quantity": request.item_quantity,
            "item_price": request.item_price,
            "total_amount": total_amount,
            "status": OrderStatus.PLACED.value,
            "events": [initial_event],
            "estimated_delivery": request.estimated_delivery,
            "actual_delivery": None,
            "delivery_delay_minutes": 0,
            "payment": payment,
            "refund": None,
            "refund_status": RefundStatus.NONE.value,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        
        # Insert order
        result = await db.orders.insert_one(order_doc)
        order_doc["_id"] = result.inserted_id
        order_doc["order_id"] = result.inserted_id  # Set order_id = _id
        
        # Update order with order_id field
        await db.orders.update_one(
            {"_id": result.inserted_id},
            {"$set": {"order_id": result.inserted_id}}
        )
        
        # Fetch the created order
        created_order = await db.orders.find_one({"_id": result.inserted_id})
        
        if not created_order:
            raise HTTPException(status_code=500, detail="Failed to retrieve created order")
        
        # Serialize and return
        serialized_order = serialize_order(created_order)
        
        log_request_end(
            logger, "POST", "/api/orders",
            status_code=201,
            duration_ms=(time.time() - start_time) * 1000,
            details={"order_id": serialized_order["order_id"]},
            user_id=request.user_id
        )
        
        return serialized_order
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "create_order_error",
            context={"user_id": request.user_id}
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/{order_id}", response_model=OrderItem)
async def update_order(order_id: str, request: UpdateOrderRequest):
    """Update an order (for inline editing)"""
    start_time = time.time()
    log_request_start(logger, "PATCH", f"/api/orders/{order_id}", body=request.model_dump())
    
    try:
        db = await get_mongodb_client()
        
        # Lookup order by order_id (handles Binary UUID, ObjectId, or string)
        order_doc = None
        
        # Approach 1: Try as UUID string and convert to Binary UUID
        if is_uuid_string(order_id):
            try:
                order_id_binary_query = uuid_to_binary(order_id)
                order_doc = await db.orders.find_one({"_id": order_id_binary_query})
            except (ValueError, TypeError):
                pass
        
        # Approach 2: If not found and looks like ObjectId (24 hex chars), try as ObjectId
        if not order_doc and len(order_id) == 24:
            try:
                if all(c in '0123456789abcdef' for c in order_id.lower()):
                    order_id_objid = ObjectId(order_id)
                    order_doc = await db.orders.find_one({"_id": order_id_objid})
            except (ValueError, TypeError):
                pass
        
        if not order_doc:
            raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
        
        # Use the found order's _id for updates
        order_id_for_query = order_doc["_id"]
        
        # Build update document
        update_doc: Dict[str, Any] = {
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        if request.status:
            update_doc["status"] = request.status
        
        if request.payment:
            update_doc["payment"] = request.payment
        
        # Update order
        result = await db.orders.update_one(
            {"_id": order_id_for_query},
            {"$set": update_doc}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
        
        # Fetch updated order
        updated_order = await db.orders.find_one({"_id": order_id_for_query})
        
        if not updated_order:
            raise HTTPException(status_code=500, detail="Failed to retrieve updated order")
        
        # Serialize and return
        serialized_order = serialize_order(updated_order)
        
        log_request_end(
            logger, "PATCH", f"/api/orders/{order_id}",
            status_code=200,
            duration_ms=(time.time() - start_time) * 1000,
            details={"order_id": serialized_order["order_id"]},
        )
        
        return serialized_order
    except HTTPException:
        raise
    except Exception as e:
        log_error_with_context(
            logger, e, "update_order_error",
            context={"order_id": order_id}
        )
        raise HTTPException(status_code=500, detail=str(e))
