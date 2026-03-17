import os

PG_HOST = os.environ.get("PG_HOST", "localhost")
PG_PORT = os.environ.get("PG_PORT", "5432")
PG_DATABASE = os.environ.get("PG_DATABASE", "face_db")
PG_USER = os.environ.get("PG_USER", "admin")
PG_PASSWORD = os.environ.get("PG_PASSWORD")
if not PG_PASSWORD:
    raise ValueError("PG_PASSWORD environment variable is required")
COLLECTION_NAME = "face_templates"
VECTOR_DIMENSION = 512