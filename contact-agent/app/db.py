from pymongo import MongoClient
from app.config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

_client = None
_db = None
_contacts = None


def get_client():
    global _client
    if _client is None:
        _client = MongoClient(MONGO_URI)
    return _client


def get_db():
    global _db
    if _db is None:
        _db = get_client()[MONGO_DB]
    return _db


def get_contacts():
    global _contacts
    if _contacts is None:
        _contacts = get_db()[MONGO_COLLECTION]
    return _contacts
