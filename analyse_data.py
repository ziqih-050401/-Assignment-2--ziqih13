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

# BALANCE TEST
treat = df[df["TREATMENT_BINARY"] == 1]
ctrl  = df[df["TREATMENT_BINARY"] == 0]

BASELINE_VARS = [
    "YEARS_EXPERIENCE",
    "BASELINE_TASKS_PER_HOUR",
    "BASELINE_ERROR_RATE",
    "TRAINING_SCORE",
]

print("=" * 80)
print("1.  BALANCE TEST — Baseline Characteristics by Group")
print("=" * 80)

balance_rows = []
for var in BASELINE_VARS:
    t_vals = treat[var].dropna()
    c_vals = ctrl[var].dropna()
    t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
    balance_rows.append({
        "Variable": var,
        "Treat Mean": round(t_vals.mean(), 3),
        "Ctrl Mean":  round(c_vals.mean(), 3),
        "Difference":  round(t_vals.mean() - c_vals.mean(), 3),
        "t-stat": round(t_stat, 3),
        "p-value": round(p_val, 4),
        "Sig (α=.05)?": "Yes *" if p_val < 0.05 else "No",
    })

balance_df = pd.DataFrame(balance_rows)
print(balance_df.to_string(index=False))

any_sig = any(r["p-value"] < 0.05 for r in balance_rows)
print("\nInterpretation:")
if not any_sig:
    print("  → No baseline variable shows a statistically significant difference")
    print("    (all p > 0.05).  Randomisation appears SUCCESSFUL.\n")
else:
    sig = [r["Variable"] for r in balance_rows if r["p-value"] < 0.05]
    print(f"  → Significant imbalance in: {', '.join(sig)}")
    print("    Consider controlling for these covariates.\n")

# Cross-tabs for categorical balance
print("Group sizes:  Treatment =", len(treat), "  Control =", len(ctrl))
print("\nSite × Treatment:")
print(pd.crosstab(df["SITE"], df["TREATMENT_BINARY"], margins=True))
print("\nQueue × Treatment:")
print(pd.crosstab(df["QUEUE"], df["TREATMENT_BINARY"], margins=True))
print("\nShift × Treatment:")
print(pd.crosstab(df["SHIFT"], df["TREATMENT_BINARY"], margins=True))

# TEST ASSUMPTIONS
print("\n" + "=" * 80)
print("2.  ASSUMPTION CHECKS")
print("=" * 80)

# 2a. Ignorability 
print("\n2a. IGNORABILITY (Conditional Independence / Unconfoundedness)")
print("-" * 60)
print("We use Welch's two-sample t-tests on every pre-treatment\n"
      "covariate to verify that treatment assignment is unrelated\n"
      "to baseline characteristics:\n")

for var in BASELINE_VARS:
    t_vals = treat[var].dropna()
    c_vals = ctrl[var].dropna()
    t_stat, p_val = stats.ttest_ind(t_vals, c_vals, equal_var=False)
    verdict = "PASS (p >= .05)" if p_val >= 0.05 else "*** FAIL ***"
    print(f"  {var:30s}  t = {t_stat:+7.3f},  p = {p_val:.4f}  →  {verdict}")

print("\nConclusion: If all tests pass, treatment assignment is independent")
print("of observed baseline characteristics — ignorability is supported.")

# 2b. SUTVA 
print("\n2b. SUTVA (Stable Unit Treatment Value Assumption)")
print("-" * 60)
print("""
  SUTVA requires that one clerk's treatment status does not affect
  another clerk's outcome.  This assumption is plausible here because:

  1. INDEPENDENT WORKSTATIONS — each clerk processes their own queue of
     loan applications on a separate terminal.  The AI pre-fill tool
     operates on each clerk's individual PDF uploads; it does not create
     a shared resource or bottleneck linking one clerk's throughput to
     another's.

  2. NO SHARED QUEUE COMPETITION — the dashboard assigns each clerk to a
     fixed loan queue (Auto, Mortgage, Personal) at a specific site.  One
     clerk finishing faster does not steal applications from neighbours or
     reduce the work pool available to others.

  3. LIMITED PEER LEARNING — the data spans a single one-week audit window
     (Feb 16–21 2026).  This short observation period limits the chance
     that control-group clerks learn AI-tool shortcuts from treated peers.

  4. GEOGRAPHIC SEPARATION — clerks are split across two physical sites
     (Irvine Ops Center and Phoenix Processing Center), further reducing
     any risk of cross-group contamination or spillover effects.

  Therefore, it is reasonable to conclude SUTVA holds in this setting.
""")

# ATE ESTIMATION
print("=" * 80)
print("3.  AVERAGE TREATMENT EFFECT (ATE) ESTIMATION")
print("=" * 80)

def estimate_ate(outcome_col, label, treat_df, ctrl_df, fmt=".2f"):
    """Compute simple difference-in-means ATE with a Welch t-test."""
    y1 = treat_df[outcome_col].dropna()
    y0 = ctrl_df[outcome_col].dropna()
    ate = y1.mean() - y0.mean()
    t_stat, p_val = stats.ttest_ind(y1, y0, equal_var=False)

    # 95 % confidence interval (Welch–Satterthwaite df)
    se = np.sqrt(y1.var()/len(y1) + y0.var()/len(y0))
    df_denom = (y1.var()/len(y1) + y0.var()/len(y0))**2 / (
        (y1.var()/len(y1))**2/(len(y1)-1) + (y0.var()/len(y0))**2/(len(y0)-1)
    )
    t_crit = stats.t.ppf(0.975, df_denom)
    ci_lo, ci_hi = ate - t_crit*se, ate + t_crit*se

    print(f"\n  {label}")
    print(f"  {''*55}")
    print(f"  Treatment mean : {y1.mean():{fmt}}   (n = {len(y1)})")
    print(f"  Control mean   : {y0.mean():{fmt}}   (n = {len(y0)})")
    print(f"  ATE            : {ate:+{fmt}}")
    print(f"  95% CI         : [{ci_lo:+{fmt}}, {ci_hi:+{fmt}}]")
    print(f"  Welch t-test   : t = {t_stat:+.3f},  p = {p_val:.4f}")
    if p_val < 0.05:
        print(f"  → Statistically SIGNIFICANT at α = 0.05")
    else:
        print(f"  → NOT significant at α = 0.05")
    return ate, t_stat, p_val, ci_lo, ci_hi

print("\n 3a. PRODUCTIVITY (Tasks Completed per Shift) ")
ate1, t1, p1, lo1, hi1 = estimate_ate("TASKS_COMPLETED",
    "Tasks Completed per Shift", treat, ctrl)

print("\n 3b. PRODUCTIVITY (Tasks per Hour) ")
ate2, t2, p2, lo2, hi2 = estimate_ate("TASKS_PER_HOUR",
    "Tasks per Hour (intensity)", treat, ctrl)

print("\n 3c. QUALITY (Error Rate %) ")
ate3, t3, p3, lo3, hi3 = estimate_ate("ERROR_RATE",
    "Error Rate (percentage points)", treat, ctrl)

#  Summary Table 
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
summary = pd.DataFrame([
    {"Outcome": "Tasks / Shift",   "ATE": f"{ate1:+.2f}", "95% CI": f"[{lo1:+.2f}, {hi1:+.2f}]",
     "t-stat": f"{t1:+.3f}", "p-value": f"{p1:.4f}",
     "Significant": "Yes" if p1 < 0.05 else "No"},
    {"Outcome": "Tasks / Hour",    "ATE": f"{ate2:+.2f}", "95% CI": f"[{lo2:+.2f}, {hi2:+.2f}]",
     "t-stat": f"{t2:+.3f}", "p-value": f"{p2:.4f}",
     "Significant": "Yes" if p2 < 0.05 else "No"},
    {"Outcome": "Error Rate (ppt)","ATE": f"{ate3:+.2f}", "95% CI": f"[{lo3:+.2f}, {hi3:+.2f}]",
     "t-stat": f"{t3:+.3f}", "p-value": f"{p3:.4f}",
     "Significant": "Yes" if p3 < 0.05 else "No"},
])
print(summary.to_string(index=False))

# Global style 
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#FAFAFA",
    "axes.edgecolor":   "#CCCCCC",
    "axes.grid":        True,
    "grid.color":       "#E8E8E8",
    "grid.linewidth":   0.6,
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   14,
    "axes.titleweight": "bold",
    "axes.labelsize":   12,
})
 
TREAT_COLOR  = "#2E86AB"   # teal-blue
CTRL_COLOR   = "#E8475F"   # coral-red
ACCENT       = "#F5A623"   # amber
 
# SCRAPE & CLEAN  (identical pipeline)
try:
    import requests
    from bs4 import BeautifulSoup
    URL = "https://bana290-assignment2.netlify.app/"
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0 (BANA290)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    rows = soup.find("table").find_all("tr")
    COLUMNS = ["CLERK","CLERK_ID","QUEUE","SITE","SHIFT",
               "YEARS_EXPERIENCE","BASELINE_TASKS_PER_HOUR","BASELINE_ERROR_RATE",
               "TRAINING_SCORE","TREATMENT","SHIFT_START","SHIFT_END",
               "TASKS_COMPLETED","ERROR_RATE"]
    records = []
    for tr in rows[1:]:
        cells = tr.find_all("td")
        if len(cells) < len(COLUMNS): continue
        records.append(dict(zip(COLUMNS, [td.get_text(" ", strip=True) for td in cells])))
    df = pd.DataFrame(records)
    print(f"[Scraper] Fetched {len(df)} rows from live site.\n")
except Exception as e:
    print(f"[Scraper] Live fetch failed ({e}); loading local CSV…\n")
    df = pd.read_csv("loan_operations_raw.csv", dtype=str, keep_default_na=False)
 
#  Treatment binary
TREAT_KW = ["AI Extract","Assist-On","Prefill Enabled","Treatment","Group A"]
CTRL_KW  = ["Control","None","Manual Entry","Typing Only","Group B"]
 
def map_treatment(x):
    if pd.isna(x): return np.nan
    x = str(x).strip()
    if x in TREAT_KW: return 1
    if x in CTRL_KW:  return 0
    raise ValueError(f"Unmapped: '{x}'")
 
df["TREATMENT_BINARY"] = df["TREATMENT"].apply(map_treatment)
df["GROUP"] = df["TREATMENT_BINARY"].map({1: "Treatment (AI)", 0: "Control (Manual)"})
 
#  Numeric cleaning 
def extract_number(text):
    if pd.isna(text): return np.nan
    t = str(text).strip()
    if t.upper() in ("TBD","--","","PENDING LOG"): return np.nan
    t = re.sub(r"/\d+", "", t)
    nums = re.sub(r"[^\d.]", "", t)
    if nums in ("","."): return np.nan
    return float(nums)
 
for c in ["YEARS_EXPERIENCE","BASELINE_TASKS_PER_HOUR","BASELINE_ERROR_RATE",
          "TRAINING_SCORE","TASKS_COMPLETED","ERROR_RATE"]:
    df[c] = df[c].apply(extract_number)
 
# Timestamps & duration 
df["SHIFT_START"] = df["SHIFT_START"].replace({"pending log": pd.NaT, "--": pd.NaT})
df["SHIFT_END"]   = df["SHIFT_END"].replace({"pending log": pd.NaT, "--": pd.NaT})
df["SHIFT_START"] = pd.to_datetime(df["SHIFT_START"], format="mixed", dayfirst=False)
df["SHIFT_END"]   = pd.to_datetime(df["SHIFT_END"],   format="mixed", dayfirst=False)
df["SHIFT_DURATION_HRS"] = ((df["SHIFT_END"]-df["SHIFT_START"]).dt.total_seconds()/3600).round(2)
df["TASKS_PER_HOUR"] = (df["TASKS_COMPLETED"]/df["SHIFT_DURATION_HRS"]).round(2)
df["CLERK"] = df["CLERK"].str.replace(r"LPC-\d+.*","",regex=True).str.strip()
 
treat = df[df["TREATMENT_BINARY"]==1]
ctrl  = df[df["TREATMENT_BINARY"]==0]
 
print(f"Clean dataset: {len(df)} rows — Treatment={len(treat)}, Control={len(ctrl)}\n")
 
# Balance Test: Baseline Covariate Comparison
BASELINE = ["YEARS_EXPERIENCE","BASELINE_TASKS_PER_HOUR",
            "BASELINE_ERROR_RATE","TRAINING_SCORE"]
LABELS   = ["Years Experience","Baseline Tasks/Hr",
            "Baseline Error Rate (%)","Training Score"]
 
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Balance Test — Baseline Covariates by Group",
             fontsize=16, fontweight="bold", y=0.97)
 
for ax, var, label in zip(axes.flat, BASELINE, LABELS):
    t_data = treat[var].dropna()
    c_data = ctrl[var].dropna()
    t_stat, p_val = stats.ttest_ind(t_data, c_data, equal_var=False)
 
    # Overlapping histograms
    bins = np.histogram_bin_edges(pd.concat([t_data, c_data]).dropna(), bins=15)
    ax.hist(t_data, bins=bins, alpha=0.55, color=TREAT_COLOR,
            label=f"Treatment (μ={t_data.mean():.2f})", edgecolor="white", linewidth=0.5)
    ax.hist(c_data, bins=bins, alpha=0.55, color=CTRL_COLOR,
            label=f"Control (μ={c_data.mean():.2f})", edgecolor="white", linewidth=0.5)
 
    # Mean lines
    ax.axvline(t_data.mean(), color=TREAT_COLOR, ls="--", lw=2)
    ax.axvline(c_data.mean(), color=CTRL_COLOR,  ls="--", lw=2)
 
    ax.set_title(f"{label}\nt={t_stat:.2f}, p={p_val:.3f}", fontsize=12)
    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.legend(fontsize=9, loc="upper right")
 
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig("plot1_balance_test.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot1_balance_test.png")
 
# Categorical Balance (Site × Queue × Shift)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Categorical Balance — Group Counts by Site, Queue & Shift",
             fontsize=15, fontweight="bold", y=1.02)
 
for ax, cat_col, title in zip(axes,
        ["SITE","QUEUE","SHIFT"],
        ["Site","Loan Queue","Shift"]):
    ct = pd.crosstab(df[cat_col], df["GROUP"])
    ct.plot(kind="bar", ax=ax, color=[CTRL_COLOR, TREAT_COLOR],
            edgecolor="white", linewidth=0.8, width=0.65)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(fontsize=9)
    # Value labels
    for container in ax.containers:
        ax.bar_label(container, fontsize=9, padding=2)
 
plt.tight_layout()
plt.savefig("plot2_categorical_balance.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot2_categorical_balance.png")
 
# ATE: Tasks Completed (Box + Strip plot)
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Average Treatment Effect — Outcome Distributions",
             fontsize=16, fontweight="bold", y=1.0)
 
palette = {"Treatment (AI)": TREAT_COLOR, "Control (Manual)": CTRL_COLOR}
 
# Tasks Completed
ax = axes[0]
sns.boxplot(data=df, x="GROUP", y="TASKS_COMPLETED", ax=ax,
            palette=palette, width=0.5, fliersize=3,
            boxprops=dict(alpha=0.7), medianprops=dict(color="black", lw=2))
sns.stripplot(data=df, x="GROUP", y="TASKS_COMPLETED", ax=ax,
              palette=palette, alpha=0.35, size=4, jitter=0.2)
ate1 = treat["TASKS_COMPLETED"].mean() - ctrl["TASKS_COMPLETED"].dropna().mean()
_, p1 = stats.ttest_ind(treat["TASKS_COMPLETED"].dropna(),
                        ctrl["TASKS_COMPLETED"].dropna(), equal_var=False)
ax.set_title(f"Tasks Completed / Shift\nATE = {ate1:+.2f}  (p < .0001)", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("Tasks Completed")
 
# Tasks per Hour
ax = axes[1]
sns.boxplot(data=df.dropna(subset=["TASKS_PER_HOUR"]),
            x="GROUP", y="TASKS_PER_HOUR", ax=ax,
            palette=palette, width=0.5, fliersize=3,
            boxprops=dict(alpha=0.7), medianprops=dict(color="black", lw=2))
sns.stripplot(data=df.dropna(subset=["TASKS_PER_HOUR"]),
              x="GROUP", y="TASKS_PER_HOUR", ax=ax,
              palette=palette, alpha=0.35, size=4, jitter=0.2)
ate2 = treat["TASKS_PER_HOUR"].mean() - ctrl["TASKS_PER_HOUR"].dropna().mean()
_, p2 = stats.ttest_ind(treat["TASKS_PER_HOUR"].dropna(),
                        ctrl["TASKS_PER_HOUR"].dropna(), equal_var=False)
ax.set_title(f"Tasks per Hour\nATE = {ate2:+.2f}  (p < .0001)", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("Tasks / Hour")
 
# Error Rate
ax = axes[2]
sns.boxplot(data=df.dropna(subset=["ERROR_RATE"]),
            x="GROUP", y="ERROR_RATE", ax=ax,
            palette=palette, width=0.5, fliersize=3,
            boxprops=dict(alpha=0.7), medianprops=dict(color="black", lw=2))
sns.stripplot(data=df.dropna(subset=["ERROR_RATE"]),
              x="GROUP", y="ERROR_RATE", ax=ax,
              palette=palette, alpha=0.35, size=4, jitter=0.2)
ate3 = treat["ERROR_RATE"].dropna().mean() - ctrl["ERROR_RATE"].dropna().mean()
_, p3 = stats.ttest_ind(treat["ERROR_RATE"].dropna(),
                        ctrl["ERROR_RATE"].dropna(), equal_var=False)
ax.set_title(f"Error Rate (%)\nATE = {ate3:+.2f} ppt  (p = {p3:.4f})", fontsize=12)
ax.set_xlabel("")
ax.set_ylabel("Error Rate (%)")
 
plt.tight_layout()
plt.savefig("plot3_ate_boxplots.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot3_ate_boxplots.png")
 
# ATE Bar Chart with Confidence Intervals
outcomes = ["Tasks / Shift", "Tasks / Hour", "Error Rate (ppt)"]
ates     = []
ci_los   = []
ci_his   = []
p_vals   = []
 
for y_col in ["TASKS_COMPLETED","TASKS_PER_HOUR","ERROR_RATE"]:
    y1 = treat[y_col].dropna()
    y0 = ctrl[y_col].dropna()
    ate = y1.mean() - y0.mean()
    se  = np.sqrt(y1.var()/len(y1) + y0.var()/len(y0))
    df_w = (y1.var()/len(y1)+y0.var()/len(y0))**2 / (
        (y1.var()/len(y1))**2/(len(y1)-1)+(y0.var()/len(y0))**2/(len(y0)-1))
    t_crit = stats.t.ppf(0.975, df_w)
    ates.append(ate)
    ci_los.append(ate - t_crit*se)
    ci_his.append(ate + t_crit*se)
    _, pv = stats.ttest_ind(y1, y0, equal_var=False)
    p_vals.append(pv)
 
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("ATE Estimates with 95% Confidence Intervals",
             fontsize=15, fontweight="bold", y=1.02)
 
colors_bar = [TREAT_COLOR, TREAT_COLOR, ACCENT]
 
for i, (ax, label) in enumerate(zip(axes, outcomes)):
    err_lo = ates[i] - ci_los[i]
    err_hi = ci_his[i] - ates[i]
    bar = ax.bar(0, ates[i], width=0.5, color=colors_bar[i], alpha=0.8,
                 edgecolor="white", linewidth=1.5)
    ax.errorbar(0, ates[i], yerr=[[err_lo],[err_hi]],
                fmt="none", ecolor="#333", elinewidth=2, capsize=8, capthick=2)
    ax.axhline(0, color="#999", lw=1, ls="-")
    ax.set_title(f"{label}\nATE = {ates[i]:+.2f}", fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Difference (Treatment − Control)")
 
    # Star annotation
    star = "***" if p_vals[i] < 0.001 else ("**" if p_vals[i] < 0.01 else
           ("*" if p_vals[i] < 0.05 else "n.s."))
    ax.annotate(star, xy=(0, ates[i] + err_hi + 0.15),
                ha="center", fontsize=14, fontweight="bold", color="#333")
 
plt.tight_layout()
plt.savefig("plot4_ate_ci_bars.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot4_ate_ci_bars.png")
 
# ATE by Subgroup (Queue type)
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("ATE by Loan Queue — Subgroup Analysis",
             fontsize=15, fontweight="bold", y=1.02)
 
for ax, (y_col, label) in zip(axes, [
        ("TASKS_COMPLETED", "Tasks / Shift"),
        ("TASKS_PER_HOUR",  "Tasks / Hour"),
        ("ERROR_RATE",      "Error Rate (%)")]):
 
    queues = sorted(df["QUEUE"].dropna().unique())
    sub_ates, sub_cis, sub_labels = [], [], []
 
    for q in queues:
        t_q = df[(df["TREATMENT_BINARY"]==1) & (df["QUEUE"]==q)][y_col].dropna()
        c_q = df[(df["TREATMENT_BINARY"]==0) & (df["QUEUE"]==q)][y_col].dropna()
        if len(t_q) < 2 or len(c_q) < 2: continue
        ate_q = t_q.mean() - c_q.mean()
        se_q  = np.sqrt(t_q.var()/len(t_q) + c_q.var()/len(c_q))
        sub_ates.append(ate_q)
        sub_cis.append(1.96 * se_q)
        sub_labels.append(q)
 
    y_pos = np.arange(len(sub_labels))
    colors_q = [TREAT_COLOR if a > 0 else CTRL_COLOR for a in sub_ates]
    ax.barh(y_pos, sub_ates, xerr=sub_cis, height=0.5,
            color=colors_q, alpha=0.8, edgecolor="white",
            capsize=5, ecolor="#555")
    ax.axvline(0, color="#999", lw=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub_labels, fontsize=11)
    ax.set_xlabel("ATE (Treatment − Control)")
    ax.set_title(label, fontsize=13, fontweight="bold")
 
    for j, (v, ci) in enumerate(zip(sub_ates, sub_cis)):
        ax.text(v + ci + 0.1 if v > 0 else v - ci - 0.1,
                j, f"{v:+.1f}", va="center",
                ha="left" if v > 0 else "right", fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig("plot5_ate_by_queue.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot5_ate_by_queue.png")
 
# Scatter: Productivity vs Error Rate (speed-accuracy)
fig, ax = plt.subplots(figsize=(9, 7))
 
for grp, color, marker, label in [
        (1, TREAT_COLOR, "o", "Treatment (AI)"),
        (0, CTRL_COLOR,  "s", "Control (Manual)")]:
    sub = df[df["TREATMENT_BINARY"]==grp].dropna(subset=["TASKS_PER_HOUR","ERROR_RATE"])
    ax.scatter(sub["TASKS_PER_HOUR"], sub["ERROR_RATE"],
               c=color, marker=marker, s=55, alpha=0.6, edgecolors="white",
               linewidth=0.5, label=label, zorder=3)
 
# Group means
for grp, color, marker in [(1, TREAT_COLOR, "o"), (0, CTRL_COLOR, "s")]:
    sub = df[df["TREATMENT_BINARY"]==grp].dropna(subset=["TASKS_PER_HOUR","ERROR_RATE"])
    ax.scatter(sub["TASKS_PER_HOUR"].mean(), sub["ERROR_RATE"].mean(),
               c=color, marker="D", s=200, edgecolors="black", linewidth=2,
               zorder=5)
 
ax.annotate("Treatment\nmean", xy=(treat["TASKS_PER_HOUR"].mean(),
            treat["ERROR_RATE"].dropna().mean()),
            xytext=(12, 4.2), fontsize=10, fontweight="bold", color=TREAT_COLOR,
            arrowprops=dict(arrowstyle="->", color=TREAT_COLOR, lw=1.5))
ax.annotate("Control\nmean", xy=(ctrl["TASKS_PER_HOUR"].dropna().mean(),
            ctrl["ERROR_RATE"].dropna().mean()),
            xytext=(7, 4.5), fontsize=10, fontweight="bold", color=CTRL_COLOR,
            arrowprops=dict(arrowstyle="->", color=CTRL_COLOR, lw=1.5))
 
ax.set_xlabel("Tasks per Hour (Productivity)")
ax.set_ylabel("Error Rate % (Quality)")
ax.set_title("Speed–Accuracy Trade-off\nAI Tool Increases Both Throughput and Errors",
             fontsize=14, fontweight="bold")
ax.legend(fontsize=11, loc="upper left")
plt.tight_layout()
plt.savefig("plot6_speed_accuracy.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ Saved plot6_speed_accuracy.png")
 
print("\n✅ All 6 plots saved successfully.")
