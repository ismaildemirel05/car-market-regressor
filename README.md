# Car Price Predictor

A machine learning project that predicts used car prices on the current French market, using real, up-to-date listing data collected from [LeBonCoin](https://www.leboncoin.fr/).

## Motivation

Existing datasets for the French used car market (data.gouv.fr, HistoVec, public Kaggle datasets) are either outdated, too limited in scope, or not representative of current retail pricing. Since car prices are highly sensitive to time (market trends, inflation, model year turnover) and geography (regional demand, urban vs. rural pricing), this project collects its own dataset directly from real, current listings rather than relying on static or historical sources.

The long-term goal is twofold:
- Build a solid end-to-end ML project (scraping → database → feature engineering → model) to showcase on GitHub.
- Use the resulting price predictions as a practical tool to spot undervalued vehicles, notably in the context of buying cars abroad and reselling them in France.

## Approach

1. **Collect data** from LeBonCoin listings using the unofficial [`lbc`](https://pypi.org/project/lbc/) Python library.
2. **Store data** incrementally in a local SQLite database, with deduplication based on listing ID.
3. **Extract and clean features** (mileage, horsepower, gearbox, fuel type, brand, model, color, doors, seats, damage status, etc.) from the raw attributes returned by LeBonCoin.
4. **Train a regression model** to predict a car's price based on its characteristics.

### A note on scraping ethics and legality

Before writing any scraping code, the legal landscape around scraping in France was researched (droit sui generis on databases, RGPD, LeBonCoin's terms of service, and Article 323-1 of the French penal code on unauthorized access to automated systems). As a result, this project follows a deliberately light and respectful scraping approach:

- Requests are spaced out (randomized delays between searches).
- `robots.txt` and the site's terms of service are respected.
- An honest User-Agent is used — no attempt to disguise the scraper as a browser.
- No IP rotation or other techniques are used to bypass anti-bot protections (e.g. DataDome), since actively circumventing a security measure would aggravate legal exposure rather than reduce it.
- Data collection is incremental (a bit every day) rather than a single large-scale scrape.

## Project structure

```
car-price-predictor/
├── config/
│   └── settings.py       # single source of truth for all configurable values
├── src/
│   ├── scraper/
│   │   ├── client.py      # LeBonCoin client wrapper
│   │   └── run_daily.py   # daily scraping entry point
│   ├── database/
│   │   ├── schema.py      # SQLite schema definition and initialization
│   │   └── repository.py  # all SQL access (insert, query, count)
│   ├── features/
│   │   └── cleaning.py    # raw listing -> clean, typed features for the model
│   └── model/              # (coming soon) training and evaluation
├── data/                   # SQLite database (not versioned)
├── logs/                   # scraper logs (not versioned)
└── requirements.txt
```

## Status

- [x] Legal and ethical research on scraping in France
- [x] Scraper (`client.py`, `run_daily.py`) collecting listings from LeBonCoin
- [x] SQLite database with dynamic schema driven by `config/settings.py`
- [x] Incremental daily collection across French regions
- [x] Feature extraction layer (mapping raw LeBonCoin attributes to clean columns)
- [x] Model training and evaluation
- [x] Prediction API

## Tech stack

- **Python**
- **[`lbc`](https://pypi.org/project/lbc/)** — unofficial LeBonCoin API client
- **SQLite** — local storage with native deduplication, well suited to incremental collection
- **pandas** — data cleaning and manipulation
- **scikit-learn** — model training

## Disclaimer

This is a personal, non-commercial learning project. Data is collected in small, spaced-out amounts, strictly for personal use and analysis. No collected data is redistributed.