"""
Daily scraping script
Manually lauch to test, and planify with Windows Task Planner for daily runs.

Usage :
    python -m src.scraper.run_daily
"""

import logging
import random
import sys
import time
from pathlib import Path

# Allowing the launch from project root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from config.settings import (
    CATEGORY,
    DB_PATH,
    DELAI_MAX_SECONDES,
    DELAI_MIN_SECONDES,
    LIMIT_PAR_RECHERCHE,
    LOG_PATH,
    PAGES_PAR_REGION,
    get_todays_regions,
)
from src.database.repository import count_annonces, insert_annonce
from src.database.schema import init_db
from src.scraper.client import get_client, search_region

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("scraper")


def run_daily_scrape():
    conn = init_db(DB_PATH)
    client = get_client()
    regions = get_todays_regions()

    logger.info("Run starting — target regions : %s", [r.name for r in regions])

    total_nouvelles = 0

    for region in regions:
        for page in range(1, PAGES_PAR_REGION + 1):
            ads = search_region(
                client=client,
                region=region,
                category=CATEGORY,
                limit=LIMIT_PAR_RECHERCHE,
                page=page,
            )

            nouvelles = 0
            for ad in ads:
                if insert_annonce(conn, ad, region.name):
                    nouvelles += 1

            total_nouvelles += nouvelles
            logger.info(
                "%s (page %d) : %d annonces reçues, %d nouvelles",
                region.name, page, len(ads), nouvelles,
            )

            # Pause entre chaque recherche, y compris entre pages
            pause = random.uniform(DELAI_MIN_SECONDES, DELAI_MAX_SECONDES)
            time.sleep(pause)

    logger.info(
        "Run ended — %d new annonces added. Total in db : %d",
        total_nouvelles, count_annonces(conn),
    )
    conn.close()


if __name__ == "__main__":
    run_daily_scrape()
