"""
Data access layer
The only part executing SQL
"""

import json
import sqlite3
from datetime import datetime, timezone


def insert_annonce(conn: sqlite3.Connection, ad, region_name: str) -> bool:
    """
    Insert an annonce if it doesn't already exists (based on id).
    Returns True if a new line has been added, else False.
    """
    attributs = {
        attr.key: attr.value
        for attr in (ad.attributes or {}).values()
    } if isinstance(ad.attributes, dict) else {}

    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO annonces
            (id, date_scrape, first_pub_date, titre, prix, marque,
             region, department, zipcode, attributs_json, url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(ad.id),
            datetime.now(timezone.utc).isoformat(),
            ad.first_publication_date,
            ad.subject,
            ad.price,
            ad.brand,
            region_name,
            ad.location.department_name if ad.location else None,
            ad.location.zipcode if ad.location else None,
            json.dumps(attributs, ensure_ascii=False),
            ad.url,
        ),
    )
    conn.commit()
    return cursor.rowcount > 0


def get_recent_annonces(conn: sqlite3.Connection, days: int = 30):
    """Returns the annonces scraped in the `days` last days."""
    cursor = conn.execute(
        """
        SELECT * FROM annonces
        WHERE date_scrape >= datetime('now', ?)
        """,
        (f"-{days} days",),
    )
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def count_annonces(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM annonces").fetchone()[0]
