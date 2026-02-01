"""UUID and ObjectId helper functions for MongoDB operations"""

from typing import Union
from uuid import UUID
from bson import Binary, ObjectId
from bson.binary import UuidRepresentation


def uuid_to_binary(uuid_string: str) -> Binary:
    """Convert UUID string to BSON Binary UUID"""
    uuid_obj = UUID(uuid_string)
    return Binary.from_uuid(uuid_obj, uuid_representation=UuidRepresentation.STANDARD)


def binary_to_uuid(binary_uuid: Union[Binary, bytes]) -> str:
    """Convert BSON Binary UUID to UUID string"""
    if isinstance(binary_uuid, bytes):
        binary_uuid = Binary(binary_uuid, subtype=4)
    return str(binary_uuid.as_uuid())


def is_uuid_string(value: str) -> bool:
    """Check if string is a valid UUID format"""
    try:
        UUID(value)
        return True
    except (ValueError, AttributeError):
        return False


def is_objectid_string(value: str) -> bool:
    """Check if string is a valid ObjectId format (24 hex characters)"""
    try:
        if isinstance(value, str) and len(value) == 24:
            # Try to create ObjectId to validate format
            ObjectId(value)
            return True
    except (ValueError, TypeError, AttributeError):
        pass
    return False


def string_to_mongo_id(id_string: str):
    """
    Convert string ID to appropriate MongoDB ID type (ObjectId or Binary UUID).
    
    Priority:
    1. If ObjectId format (24 hex chars) -> ObjectId
    2. If UUID format -> Binary UUID
    3. Otherwise -> return as-is (string)
    """
    if is_objectid_string(id_string):
        return ObjectId(id_string)
    elif is_uuid_string(id_string):
        return uuid_to_binary(id_string)
    else:
        return id_string
