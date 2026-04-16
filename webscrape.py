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

