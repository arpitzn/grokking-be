"""UUID helper functions for MongoDB operations"""

from typing import Union
from uuid import UUID
from bson import Binary
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
