"""
Project config
Contains all the values needed to execute
"""

from datetime import datetime
from lbc.model.enums import Region, Category

# --- Paths ---
DB_PATH = "data/annonces_voitures.db"
LOG_PATH = "logs/scraper.log"

# --- Car category ---
CATEGORY = Category.VEHICULES_VOITURES

# --- Regions (13 métropolitaines) ---
# Volontarily dropped the older regions to not duplicate entries
REGIONS_ACTUELLES = [
    Region.ILE_DE_FRANCE,
    Region.AUVERGNE_RHONE_ALPES,
    Region.BOURGOGNE_FRANCHE_COMTE,
    Region.BRETAGNE,
    Region.CENTRE_VAL_DE_LOIRE,
    Region.CORSE,
    Region.GRAND_EST,
    Region.HAUTS_DE_FRANCE,
    Region.NORMANDIE,
    Region.NOUVELLE_AQUITAINE,
    Region.OCCITANIE,
    Region.PAYS_DE_LA_LOIRE,
    Region.PROVENCE_ALPES_COTE_DAZUR,
]

# --- Scraping parameters (adapt if you're blocked) ---
REGIONS_PAR_JOUR = 6          # nb of regions explored daily
LIMIT_PAR_RECHERCHE = 50      # nb annonces per search
PAGES_PAR_REGION = 1          # nb pages per region and per run

DELAI_MIN_SECONDES = 8        # min pause between searchs
DELAI_MAX_SECONDES = 20       # max pause between searchs


def get_todays_regions(n: int = REGIONS_PAR_JOUR) -> list[Region]:
    """
    Selects n different regions based on datetime,
    to make sure to visit every region over a few days
    rather than searching for everything instantly.
    """
    jour = datetime.now().timetuple().tm_yday
    start = (jour * n) % len(REGIONS_ACTUELLES)
    return [REGIONS_ACTUELLES[(start + i) % len(REGIONS_ACTUELLES)] for i in range(n)]
