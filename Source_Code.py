"""
Project 4 - Air Quality Insights Dashboard
Script 1: Load, inspect and clean the dataset  (Tasks 49-52)

Input : air_quality_dirty_dataset.csv   (raw)
Output: air_quality_clean.csv           (cleaned, used by 02_analysis.py)
        cleaning_log.txt                (what was changed and why)
"""
import re
import numpy as np
import pandas as pd

RAW_FILE = "air_quality_dirty_dataset.csv"
CLEAN_FILE = "air_quality_clean.csv"
LOG_FILE = "cleaning_log.txt"

# Plausible physical ranges. Values outside these are treated as bad sensor/entry data.
VALID_RANGE = {
    "PM25": (0, 1000),            # ug/m3  (-999 is a "no data" placeholder)
    "PM10": (0, 1000),            # ug/m3
    "NO2": (0, 500),              # ug/m3
    "Temperature_C": (-10, 55),   # deg C, plausible for these North-Indian cities
    "Humidity_Percent": (0, 100), # percent
}
NUM_COLS = list(VALID_RANGE)
log = []


def note(msg):
    print(msg)
    log.append(msg)


# ---------------------------------------------------------------- Task 49
df = pd.read_csv(RAW_FILE)
note(f"[49] Loaded {RAW_FILE}: {df.shape[0]} rows x {df.shape[1]} columns")
print(df.head(), "\n")
print(df.dtypes, "\n")

# ---------------------------------------------------------------- Task 50
missing = df.isna().sum()
note("[50] Missing values per column (raw):\n" + missing.to_string())

# ---------------------------------------------------------------- Cleaning
# 1. City: trim spaces and fix capitalisation (' Delhi ', 'DELHI', 'delhi' -> 'Delhi')
df["City"] = df["City"].str.strip().str.title()
note(f"[clean] City names standardised -> {df['City'].nunique()} cities: {sorted(df['City'].dropna().unique())}")

# 2. Date: three formats exist (YYYY-MM-DD, YYYY.MM.DD, DD/MM/YYYY).
#    In the DD/MM/YYYY rows the first number exceeds 12 in 20 cases and the
#    second never does, so the format is day-first, not month-first.
def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    s = s.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return pd.to_datetime(s, format="%Y-%m-%d")
    if re.fullmatch(r"\d{4}\.\d{2}\.\d{2}", s):
        return pd.to_datetime(s, format="%Y.%m.%d")
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", s):
        return pd.to_datetime(s, format="%d/%m/%Y")
    return pd.NaT

# ---------------------------------------------------------------- Task 52
df["Date"] = df["Date"].apply(parse_date)
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
note("[52] Date converted to datetime64 and Month / Month_Name extracted")

# 3. Rows with no usable Date or City cannot be placed in time or space
before = len(df)
df = df.dropna(subset=["Date", "City"])
note(f"[clean] Dropped {before - len(df)} rows with missing Date or City")
df["Month"] = df["Month"].astype(int)   # whole numbers (1-12) once NaT rows are gone

# 4. Exact duplicates (checked after standardising City/Date)
before = len(df)
df = df.drop_duplicates()
note(f"[clean] Dropped {before - len(df)} exact duplicate rows")

# 5. Placeholders and impossible values -> NaN
for col, (lo, hi) in VALID_RANGE.items():
    bad = (df[col] < lo) | (df[col] > hi)
    note(f"[clean] {col}: {int(bad.sum())} values outside [{lo}, {hi}] set to NaN "
         f"(includes {(df[col] == -999).sum()} placeholder -999)")
    df.loc[bad, col] = np.nan

# 6. Flag dates later than "today" (kept, not deleted - decision left to the analyst)
TODAY = pd.Timestamp.today().normalize()
df["Future_Date"] = df["Date"] > TODAY
note(f"[clean] {int(df['Future_Date'].sum())} rows have a Date after {TODAY.date()} "
     "(flagged in column Future_Date, kept)")

# ---------------------------------------------------------------- Task 51
# PM2.5 imputation: median of the same City and Month; if that group has no
# valid reading, fall back to the City median.
# Why median: PM2.5 is right-skewed with extreme spikes, so the mean would be
# pulled upward. Why city+month: pollution differs by place and season.
# Why not drop: rows still hold valid PM10/NO2/Temperature data.
df["PM25_Imputed"] = df["PM25"].isna()
n_missing = int(df["PM25_Imputed"].sum())
city_month_med = df.groupby(["City", "Month"])["PM25"].transform("median")
city_med = df.groupby("City")["PM25"].transform("median")
df["PM25"] = df["PM25"].fillna(city_month_med).fillna(city_med)
note(f"[51] Imputed {n_missing} missing/invalid PM2.5 values with City-Month median "
     f"(City-median fallback). Remaining missing PM2.5: {int(df['PM25'].isna().sum())}")

# ---------------------------------------------------------------- Final checks
df = df.sort_values("Date").reset_index(drop=True)
note("[final] Missing values after cleaning:\n" + df.isna().sum().to_string())
note(f"[final] Clean shape: {df.shape}")
df.to_csv(CLEAN_FILE, index=False)
with open(LOG_FILE, "w") as f:
    f.write("\n\n".join(log))
print(f"\nSaved {CLEAN_FILE} and {LOG_FILE}")
