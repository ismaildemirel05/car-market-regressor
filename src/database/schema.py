"""
SQLite db schema definition and initialization
"""

import sqlite3


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Opens a connexion to the db and ensure it exists
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annonces (
            id              TEXT PRIMARY KEY,
            scrape_date     TEXT NOT NULL,
            first_pub_date  TEXT,
            title           TEXT,
            price           REAL,
            brand           TEXT,
            region          TEXT,
            department      TEXT,
            zipcode         TEXT,
            attributes_json TEXT,
            url             TEXT
        )
        """
    )
    conn.commit()
    return conn
