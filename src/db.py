import logging
import sqlite3
import psycopg2
from config import DATABASE_FILE
from database_config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

logger = logging.getLogger(__name__)


def get_metadata_connection() -> sqlite3.Connection:
    """Create and return a SQLite metadata database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    logger.info("Connected to metadata database: %s", DATABASE_FILE)
    return conn


def get_vector_connection() -> psycopg2.extensions.connection:
    """Create and return a PostgreSQL pgvector database connection."""
    conn = psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        database=PG_DATABASE,
        user=PG_USER,
        password=PG_PASSWORD
    )
    conn.autocommit = True
    logger.info("Connected to vector database: %s@%s:%s", PG_DATABASE, PG_HOST, PG_PORT)
    return conn
