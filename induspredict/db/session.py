"""
IndusPredict AI - Database Engine, Session Management & Backups.
Configures SQLite with Write-Ahead Logging (WAL) and foreign key enforcement.
"""

from contextlib import contextmanager
from datetime import datetime
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker

from induspredict.core.config import APP_DATA_DIR, BACKUP_DIR, DATABASE_URL, DEFAULT_DB_FILE
from induspredict.core.logger import get_logger
from induspredict.db.models import Base

logger = get_logger("db.session")

# SQLite connection pragmas for high concurrency and relational integrity
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
ScopedSession = scoped_session(SessionFactory)


def init_db(target_engine=None) -> None:
    """Create all database tables."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    logger.info("Database tables initialized successfully.")


@contextmanager
def get_db_session() -> Generator:
    """Context manager for thread-safe database sessions with rollback on error."""
    session = ScopedSession()
    try:
        yield session
        session.commit()
    except Exception as exc:
        session.rollback()
        logger.error(f"Database transaction rolled back due to error: {exc}")
        raise
    finally:
        session.close()


def backup_database(destination_path: Optional[Path] = None) -> Path:
    """
    Perform hot SQLite backup using the native sqlite3 online backup API.
    Guarantees consistent snapshots even during active writes.
    """
    if destination_path is None:
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        destination_path = BACKUP_DIR / f"induspredict_backup_{timestamp_str}.db"

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    # Use SQLite online backup API
    src_conn = sqlite3.connect(str(DEFAULT_DB_FILE))
    dest_conn = sqlite3.connect(str(destination_path))
    try:
        with dest_conn:
            src_conn.backup(dest_conn, pages=100)
        logger.info(f"Database hot backup completed successfully to: {destination_path}")
    finally:
        dest_conn.close()
        src_conn.close()

    return destination_path
