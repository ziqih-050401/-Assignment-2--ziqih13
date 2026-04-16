import re, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 140)

# Scrape the HTML table
try:
    import requests
    from bs4 import BeautifulSoup

    URL = "https://bana290-assignment2.netlify.app/"
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (BANA290)"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = soup.find("table")
    rows = table.find_all("tr")

    COLUMNS = [
        "CLERK","CLERK_ID","QUEUE","SITE","SHIFT",
        "YEARS_EXPERIENCE","BASELINE_TASKS_PER_HOUR","BASELINE_ERROR_RATE",
        "TRAINING_SCORE","TREATMENT","SHIFT_START","SHIFT_END",
        "TASKS_COMPLETED","ERROR_RATE",
    ]

    records = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < len(COLUMNS):
            continue
        records.append(dict(zip(COLUMNS,
                        [td.get_text(" ", strip=True) for td in cells])))

    df = pd.DataFrame(records)
    print(f"[Scraper]  Fetched {len(df)} rows from live site.\n")

except Exception as e:
    print(f"[Scraper]  Live fetch failed ({e}); loading from local CSV...\n")
    df = pd.read_csv("loan_operations_raw.csv", dtype=str, keep_default_na=False)

