"""
Pytest configuration and fixtures for IndusPredict AI.
Sets up isolated temporary databases and offscreen Qt environments.
"""

import os
from pathlib import Path
import tempfile
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

# Configure offscreen Qt platform for headless automated GUI testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp()

from induspredict.db.models import Base
from induspredict.db.seed import seed_database
import induspredict.db.session as session_module


@pytest.fixture(scope="session")
def test_db_path():
    """Create a temporary SQLite database file for testing."""
    tmp_dir = tempfile.mkdtemp()
    db_file = Path(tmp_dir) / "test_induspredict.db"
    return db_file


@pytest.fixture(scope="session")
def test_engine(test_db_path):
    """Engine connected to the temporary test database."""
    test_db_url = f"sqlite:///{test_db_path}"
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    # Monkeypatch the engine and sessionmaker in the session module
    session_module.engine = engine
    session_module.SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session_module.ScopedSession = scoped_session(session_module.SessionFactory)

    # Seed the test database
    seed_database(force=True)

    yield engine

    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Provide a transactional database session for each test."""
    session = session_module.ScopedSession()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="session")
def qapp():
    """Provide a headless QApplication instance for GUI tests."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if not app:
        app = QApplication([])
    return app
