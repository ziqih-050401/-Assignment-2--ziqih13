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

#  Clean TREATMENT → binary 1 / 0 
TREATMENT_KEYWORDS = ["AI Extract", "Assist-On", "Prefill Enabled",
                      "Treatment", "Group A"]
CONTROL_KEYWORDS   = ["Control", "None", "Manual Entry",
                      "Typing Only", "Group B"]

def map_treatment(raw_label) -> int:
    """Check which list a row's value falls into → 1 (treatment) or 0 (control)."""
    if pd.isna(raw_label):
        return np.nan
    label = str(raw_label).strip()
    if label in TREATMENT_KEYWORDS:
        return 1
    elif label in CONTROL_KEYWORDS:
        return 0
    else:
        raise ValueError(f"Unmapped treatment label: '{label}'")

df["TREATMENT_BINARY"] = df["TREATMENT"].apply(map_treatment)

#  Clean numeric columns using regex 
def extract_number(text: str) -> float:
    """
    Use regex to replace everything except digits and decimal points,
    then convert to float.  Returns NaN for 'TBD', '--', etc.
    """
    if pd.isna(text):
        return np.nan
    cleaned = text.strip()
    if cleaned.upper() in ("TBD", "--", "", "PENDING LOG"):
        return np.nan
    # Handle "85/100" → take numerator only
    cleaned = re.sub(r"/\d+", "", cleaned)
    nums = re.sub(r"[^\d.]", "", cleaned)       # <-- core regex
    if nums in ("", "."):
        return np.nan
    return float(nums)

NUMERIC_COLS = [
    "YEARS_EXPERIENCE", "BASELINE_TASKS_PER_HOUR", "BASELINE_ERROR_RATE",
    "TRAINING_SCORE", "TASKS_COMPLETED", "ERROR_RATE",
]
for col in NUMERIC_COLS:
    df[col] = df[col].apply(extract_number)

#  Standardise timestamps & compute shift duration 
df["SHIFT_START"] = df["SHIFT_START"].replace({"pending log": pd.NaT, "--": pd.NaT})
df["SHIFT_END"]   = df["SHIFT_END"].replace({"pending log": pd.NaT, "--": pd.NaT})

df["SHIFT_START"] = pd.to_datetime(df["SHIFT_START"], format="mixed", dayfirst=False)
df["SHIFT_END"]   = pd.to_datetime(df["SHIFT_END"],   format="mixed", dayfirst=False)

df["SHIFT_DURATION_HRS"] = (
    (df["SHIFT_END"] - df["SHIFT_START"]).dt.total_seconds() / 3600
).round(2)

#  Additional cleaning 
df["CLERK"] = df["CLERK"].str.replace(r"LPC-\d+.*", "", regex=True).str.strip()
df["TASKS_PER_HOUR"] = (df["TASKS_COMPLETED"] / df["SHIFT_DURATION_HRS"]).round(2)
df["QUEUE"] = df["QUEUE"].astype("category")
df["SITE"]  = df["SITE"].astype("category")
df["SHIFT"] = df["SHIFT"].astype("category")

print(f"Clean dataset: {len(df)} rows,  "
      f"Treatment(1)={df['TREATMENT_BINARY'].sum()},  "
      f"Control(0)={(df['TREATMENT_BINARY']==0).sum()}\n")
