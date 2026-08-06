import lbc
from lbc import Region
import json
from pathlib import Path
import pandas
from dataclasses import fields, is_dataclass


PROPERTIES = [
    "id", 
    "first_publication_date", 
    "expiration_date",
    "category_name",
    "price",
    "brand",
    "model",
    "regdate",
    "mileage",
    "fuel",
    "gearbox",
    "doors",
    "seats",
    "vehicle_technical_inspection_a",
    "issuance_date",
    "vehicle_damage",
    "vehicle_type",
    "vehicule_color",
    "horsepower",
    "horse_power_din",
    "vehicle_vsp",
    "car_price_min",
    "car_price_max",
    "region_id",
    "department_id",
    "city",
]


#########################################
# Research
#########################################

def search_cars(client: lbc.Client, nb_results: int) -> lbc.Search:
    return client.search(
            text="voiture",
            locations=REGIONS,
            page=1,
            limit=nb_results,
            sort=lbc.Sort.NEWEST,
            ad_type=lbc.AdType.OFFER,
            category=lbc.Category.VEHICULES_VOITURES,
            price=[0,20_000]
        )


#########################################
# Writing data
#########################################

def to_plain(obj):
    if is_dataclass(obj):
        return {
            f.name: to_plain(getattr(obj, f.name))
            for f in fields(obj)
            if not f.name.startswith("_")
        }
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def write_json(data: dict, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_result_json(result: lbc.Search, out_path: Path):
    write_json(to_plain(result), DATA_DIR / "data.json")

    

#########################################
# Main
#########################################

REGIONS = [
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


DATA_DIR = Path(r".\data")

def main():
    client = lbc.Client()
    nb_results = 50
    result = search_cars(client, nb_results=nb_results)

    json_out_path =  DATA_DIR / "data.json"
    write_result_json(result, json_out_path)


if __name__ == "__main__":
    main()