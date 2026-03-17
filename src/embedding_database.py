import logging
import psycopg2
from psycopg2 import sql
from database_config import PG_DATABASE, PG_HOST, PG_PASSWORD, PG_PORT, PG_USER, COLLECTION_NAME, VECTOR_DIMENSION

logger = logging.getLogger(__name__)


class PgVectorDBSetup:
    def __init__(self):
        self.conn = None
        try:
            self.conn = psycopg2.connect(
                host=PG_HOST,
                port=PG_PORT,
                database=PG_DATABASE,
                user=PG_USER,
                password=PG_PASSWORD
            )
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL DB '%s'", PG_DATABASE)
        except psycopg2.OperationalError as e:
            logger.error("Cannot connect to PostgreSQL: %s", e)
            raise RuntimeError("Cannot connect to PostgreSQL.")
        except Exception as e:
            logger.error("Unknown error connecting to DB: %s", e)
            raise RuntimeError("Cannot connect to PostgreSQL.")

    def _execute_sql(self, query: sql.SQL, *args):
        """Execute SQL safely."""
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, args)
                return True
        except psycopg2.Error as e:
            logger.error("SQL execution error: %s", e)
            return False

    def setup_pgvector(self):
        """Ensure pgvector extension is enabled."""
        logger.info("Checking and enabling 'vector' extension...")
        create_extension_query = sql.SQL("CREATE EXTENSION IF NOT EXISTS vector;")
        if self._execute_sql(create_extension_query):
            logger.info("Extension 'vector' is ready")
        else:
            logger.critical("Cannot create 'vector' extension. Ensure pgvector is installed.")
            raise RuntimeError("pgvector installation required.")

    def create_face_collection(self):
        """Create 'face_templates' table with 512-dim vector column."""
        create_table_query = sql.SQL("""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id VARCHAR(100) UNIQUE NOT NULL,
                enrollment_date BIGINT NOT NULL,
                face_vector vector({dim})
            );
        """).format(
            table_name=sql.Identifier(COLLECTION_NAME),
            dim=sql.Literal(VECTOR_DIMENSION)
        )

        logger.info("Creating or verifying table '%s'...", COLLECTION_NAME)
        if self._execute_sql(create_table_query):
            logger.info("Table '%s' ready", COLLECTION_NAME)
        else:
            logger.error("Cannot create table '%s'", COLLECTION_NAME)
            return

        index_query = sql.SQL("""
            CREATE INDEX IF NOT EXISTS face_vector_ivfflat_idx
            ON {table_name} USING ivfflat (face_vector) WITH (lists = 100);
        """).format(table_name=sql.Identifier(COLLECTION_NAME))

        logger.info("Creating IVFFlat index on vector column...")
        if self._execute_sql(index_query):
            logger.info("IVFFlat index created on 'face_vector'")
        else:
            logger.warning("Cannot create index. Search performance may be affected.")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed PostgreSQL connection")


def main():
    """Run pgvector database setup."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    logger.info("--- EMBEDDING DATABASE SETUP (POSTGRESQL + PGVECTOR) ---")
    setup = None
    try:
        setup = PgVectorDBSetup()
        setup.setup_pgvector()
        setup.create_face_collection()
        logger.info("SETUP COMPLETE: PostgreSQL + pgvector ready for 512-dim face vectors")
    except RuntimeError:
        logger.error("SETUP FAILED: Connection or configuration error")
    except Exception as e:
        logger.exception("SETUP FAILED: Unknown error: %s", e)
    finally:
        if setup:
            setup.close()


if __name__ == "__main__":
    main()
