"""
IndusPredict AI - Intelligent Predictive Maintenance System.
Application Entry Point.
"""

import os
import sys
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from induspredict.core.config import AppConfig
from induspredict.core.logger import get_logger
from induspredict.db.seed import seed_database
from induspredict.db.session import init_db
from induspredict.ui.main_window import MainWindow

logger = get_logger("app.main")


def main() -> None:
    """Initialize database, seed sample assets, and start PySide6 desktop GUI."""
    logger.info("Initializing IndusPredict AI application...")

    # Initialize SQLite database and seed initial assets
    try:
        init_db()
        seed_database(force=False)
    except Exception as e:
        logger.error(f"Critical error initializing database: {e}", exc_info=True)

    # Initialize Qt Application
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)

    app.setApplicationName(AppConfig.app_name)
    app.setApplicationVersion(AppConfig.version)
    app.setOrganizationName(AppConfig.organization)

    # Launch Main Window
    window = MainWindow()
    window.show()

    logger.info("IndusPredict AI desktop application launched successfully.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
