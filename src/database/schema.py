"""
SQLite db schema definition and initialization
"""

import sqlite3


def init_db(db_path: str) -> sqlite3.Connection:
    """
    Ouvre une connexion à la base et s'assure que la table existe.
    Sans danger d'appeler cette fonction à chaque run (CREATE TABLE IF NOT EXISTS).
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS annonces (
            id              TEXT PRIMARY KEY,
            date_scrape     TEXT NOT NULL,
            first_pub_date  TEXT,
            titre           TEXT,
            prix            REAL,
            marque          TEXT,
            region          TEXT,
            department      TEXT,
            zipcode         TEXT,
            attributs_json  TEXT,
            url             TEXT
        )
        """
    )
    conn.commit()
    return conn
