import logging
from pymongo.errors import PyMongoError
from app.db import get_contacts
from app.embeddings import embed_contact

logger = logging.getLogger("sync")
logging.basicConfig(level=logging.INFO)


def upsert_embedding(doc_id):
    doc = get_contacts().find_one({"_id": doc_id})
    if not doc:
        return
    vector = embed_contact(doc)
    get_contacts().update_one({"_id": doc_id}, {"$set": {"embedding": vector}})
    logger.info(f"Embedded contact {doc_id}")


def backfill_all():
    """Run once (e.g. via `python -m app.sync backfill`) to embed all existing docs."""
    cursor = get_contacts().find({"embedding": {"$exists": False}})
    count = 0
    for doc in cursor:
        vector = embed_contact(doc)
        get_contacts().update_one({"_id": doc["_id"]}, {"$set": {"embedding": vector}})
        count += 1
    logger.info(f"Backfilled embeddings for {count} contacts")
    return count


def start_change_stream_listener():
    """
    Watches the contacts collection for insert/update/delete and keeps
    the 'embedding' field in sync automatically. No manual rebuild needed.

    Requires MongoDB to run as a replica set (Atlas clusters are replica
    sets by default; a standalone local mongod is not).
    """
    try:
        with get_contacts().watch(full_document="updateLookup") as stream:
            logger.info("Change stream listener started — watching for inserts/updates/deletes.")
            for change in stream:
                op = change["operationType"]
                try:
                    if op == "insert":
                        doc = change["fullDocument"]
                        # avoid re-triggering on our own embedding writes
                        if "embedding" not in doc:
                            upsert_embedding(doc["_id"])

                    elif op == "update":
                        updated_fields = change.get("updateDescription", {}).get("updatedFields", {})
                        if "embedding" in updated_fields:
                            continue  # this update WAS the embedding write, skip
                        doc_id = change["documentKey"]["_id"]
                        upsert_embedding(doc_id)

                    elif op == "delete":
                        # Nothing to clean up separately — the doc (and its
                        # embedding field) is already gone from the same collection.
                        logger.info(f"Contact deleted: {change['documentKey']['_id']}")

                except PyMongoError as e:
                    logger.error(f"Error processing change event: {e}")

    except PyMongoError as e:
        logger.warning(
            "Change stream not available (needs MongoDB replica set, e.g. Atlas). "
            f"Falling back to manual sync only. Error: {e}"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        backfill_all()
    else:
        print("Usage: python -m app.sync backfill")
