"""
Manages the client
Customize the client if needed (proxy, logging, ...)
"""

import logging

import lbc
from lbc.exceptions import DatadomeError, RequestError

logger = logging.getLogger("scraper")


def get_client() -> lbc.Client:
    return lbc.Client()


def search_region(client: lbc.Client, region, category, limit: int, page: int = 1):
    """
    Does a research for a given region
    Returns the list of annonces or an empty list in case of error
    (no aggressive retry)
    """
    try:
        result = client.search(
            category=category,
            locations=[region],
            limit=limit,
            page=page,
            sort=lbc.Sort.NEWEST,
            ad_type=lbc.AdType.OFFER,
        )
        return result.ads

    except DatadomeError:
        logger.warning(
            "Blocked by DataDome on %s — run stopping, retry later.",
            region.name,
        )
        return []

    except RequestError as e:
        logger.warning("Request error on %s : %s", region.name, e)
        return []
