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
    Fait une recherche pour une région donnée.
    Retourne la liste d'annonces, ou une liste vide en cas de blocage/erreur
    (on ne fait jamais de retry agressif ici : un blocage doit ralentir le
    scraper, pas déclencher une nouvelle tentative immédiate).
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
            "Bloqué par DataDome sur %s — arrêt du run, réessaie plus tard.",
            region.name,
        )
        return []

    except RequestError as e:
        logger.warning("Erreur de requête sur %s : %s", region.name, e)
        return []
