import lbc
import json
from pathlib import Path
import pandas
from dataclasses import fields, is_dataclass


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
            price=[0,100_000]
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



#########################################
# Main
#########################################

REGIONS = list(lbc.model.Region)
DATA_DIR = Path(r".\data")

def main():
    client = lbc.Client()
    nb_results = 10
    result = search_cars(client, nb_results=nb_results)
    write_json(to_plain(result), DATA_DIR / "data.json")


if __name__ == "__main__":
    main()