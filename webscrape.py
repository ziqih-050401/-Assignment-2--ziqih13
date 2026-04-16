import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

# 1.  SCRAPE – fetch page and extract the raw table
URL = "https://bana290-assignment2.netlify.app/"
response = requests.get(URL, headers={
    "User-Agent": "Mozilla/5.0 (compatible; BANA290-scraper/1.0)"
})
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")
rows  = table.find_all("tr")

# Canonical column names (match the header row in the HTML)
COLUMNS = [
    "CLERK", "CLERK_ID", "QUEUE", "SITE", "SHIFT",
    "YEARS_EXPERIENCE", "BASELINE_TASKS_PER_HOUR", "BASELINE_ERROR_RATE",
    "TRAINING_SCORE", "TREATMENT", "SHIFT_START", "SHIFT_END",
    "TASKS_COMPLETED", "ERROR_RATE",
]

records = []
for tr in rows[1:]:                           # skip header row
    cells = tr.find_all("td")
    if len(cells) < len(COLUMNS):
        continue
    raw = [td.get_text(" ", strip=True) for td in cells]
    records.append(dict(zip(COLUMNS, raw)))

df = pd.DataFrame(records)
print(f"Scraped {len(df)} rows  x  {len(df.columns)} columns\n")

# 2.  CLEAN TREATMENT → binary 1 / 0
# Two predefined keyword lists
TREATMENT_KEYWORDS = [
    "AI Extract", "Assist-On", "Prefill Enabled",
    "Treatment", "Group A",
]
CONTROL_KEYWORDS = [
    "Control", "None", "Manual Entry",
    "Typing Only", "Group B",
]

def map_treatment(raw_label: str) -> int:
    """Return 1 for Treatment group, 0 for Control group."""
    label = raw_label.strip()
    if label in TREATMENT_KEYWORDS:
        return 1
    elif label in CONTROL_KEYWORDS:
        return 0
    else:
        raise ValueError(f"Unmapped treatment label: '{label}'")

df["TREATMENT_BINARY"] = df["TREATMENT"].apply(map_treatment)

# 3.  CLEAN NUMERIC COLUMNS  (regex approach)
def extract_number(text: str) -> float:
    """
    Strip everything except digits and decimal points, then convert
    to float.  Returns NaN for placeholders like 'TBD', '--', etc.
    """
    if pd.isna(text):
        return float("nan")
    cleaned = text.strip()
    # Catch known missing-data markers first
    if cleaned.upper() in ("TBD", "--", "", "PENDING LOG"):
        return float("nan")
    # Keep only digits and the decimal point
    nums = re.sub(r"[^\d.]", "", cleaned)
    if nums == "" or nums == ".":
        return float("nan")
    return float(nums)

numeric_cols = [
    "YEARS_EXPERIENCE",
    "BASELINE_TASKS_PER_HOUR",
    "BASELINE_ERROR_RATE",
    "TRAINING_SCORE",
    "TASKS_COMPLETED",
    "ERROR_RATE",
]

for col in numeric_cols:
    df[col] = df[col].apply(extract_number)

# 4.  CLEAN & STANDARDISE TIMESTAMPS  →  compute SHIFT_DURATION_HRS
# Replace sentinel strings with NaN before parsing
df["SHIFT_START"] = df["SHIFT_START"].replace(
    {"pending log": pd.NaT, "--": pd.NaT}
)
df["SHIFT_END"] = df["SHIFT_END"].replace(
    {"pending log": pd.NaT, "--": pd.NaT}
)

# pd.to_datetime with infer_datetime_format handles all the mixed
# formats in the dashboard:
#   "2026-02-18 15:50"   (ISO)
#   "Feb 18, 2026 07:56" (Month-name)
#   "21-Feb-2026 08:19 AM" (DD-Mon-YYYY with AM/PM)
#   "02/21/2026 04:26 PM"  (US-style with AM/PM)
df["SHIFT_START"] = pd.to_datetime(df["SHIFT_START"], format="mixed", dayfirst=False)
df["SHIFT_END"]   = pd.to_datetime(df["SHIFT_END"],   format="mixed", dayfirst=False)

# Shift duration in hours
df["SHIFT_DURATION_HRS"] = (
    (df["SHIFT_END"] - df["SHIFT_START"]).dt.total_seconds() / 3600
).round(2)

# 5.  ADDITIONAL CLEANING FOR ANALYSIS READINESS

# 5a. Clean the CLERK name  (the first cell bundles "Leah Kim LPC-2125
#     · Irvine Ops Center" — keep only the name portion)
df["CLERK"] = (
    df["CLERK"]
    .str.replace(r"LPC-\d+.*", "", regex=True)   # drop everything from LPC-…
    .str.strip()
)

# 5b. Derive TASKS_PER_HOUR from cleaned fields
df["TASKS_PER_HOUR"] = (
    df["TASKS_COMPLETED"] / df["SHIFT_DURATION_HRS"]
).round(2)

# 5c. Confirm dtypes
df["QUEUE"] = df["QUEUE"].astype("category")
df["SITE"]  = df["SITE"].astype("category")
df["SHIFT"] = df["SHIFT"].astype("category")

# 6.  INSPECT THE RESULT
print("── dtypes ──")
print(df.dtypes, "\n")

print("── Missing values ──")
print(df.isnull().sum(), "\n")

print("── Treatment distribution ──")
print(df["TREATMENT_BINARY"].value_counts().rename({1: "Treatment (1)", 0: "Control (0)"}), "\n")

print("── Numeric summary ──")
print(df[numeric_cols + ["SHIFT_DURATION_HRS", "TASKS_PER_HOUR"]].describe().round(2), "\n")

print("── First 5 rows ──")
print(df.head().to_string(), "\n")

# 7.  EXPORT
OUTPUT = "loan_operations_clean.csv"
df.to_csv(OUTPUT, index=False)
print(f"Saved cleaned dataset → '{OUTPUT}'  ({len(df)} rows)")
 