"""
Project 4 - Air Quality Insights Dashboard
Script 1: Load, inspect and clean the dataset  (Tasks 49-52)

Input : air_quality_dirty_dataset.csv   (raw)
Output: air_quality_clean.csv           (cleaned, used by 02_analysis.py)
        cleaning_log.txt                (what was changed and why)
"""
import numpy as np
import pandas as pd


# `df` is the single working variable. The log lives in its metadata.
df = pd.read_csv("air_quality_clean.csv")
df.attrs["log"] = [
    f"[49] Loaded air_quality_dirty_dataset.csv: {df.shape[0]} rows x {df.shape[1]} columns",
    "[50] Missing values per column (raw):\n" + df.isna().sum().to_string(),
]
print(df.head(), "\n")
print(df.dtypes, "\n")

# City: trim spaces and standardise capitalisation.
df["City"] = df["City"].astype("string").str.strip().str.title()
df.attrs["log"].append(
    f"[clean] City names standardised -> {df['City'].nunique()} cities: "
    f"{sorted(df['City'].dropna().unique())}"
)

# Date: parse the three known formats without a custom parser or `re`.
df["Date"] = df["Date"].astype("string").str.strip()
df["Date"] = (
    pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    .fillna(pd.to_datetime(df["Date"], format="%Y.%m.%d", errors="coerce"))
    .fillna(pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce"))
)
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
df.attrs["log"].append("[52] Date converted to datetime64 and Month / Month_Name extracted")

# Rows with no usable Date or City cannot be placed in time or space.
df.attrs["log"].append(
    f"[clean] Dropped {int(df[['Date', 'City']].isna().any(axis=1).sum())} rows with missing Date or City"
)
df = df.dropna(subset=["Date", "City"])
df["Month"] = df["Month"].astype(int)

# Exact duplicates are checked after standardising City and Date.
df.attrs["log"].append(f"[clean] Dropped {int(df.duplicated().sum())} exact duplicate rows")
df = df.drop_duplicates()

# Convert measurements to numbers, then replace impossible sensor values with NaN.
df[["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]] = df[
    ["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]
].apply(pd.to_numeric, errors="coerce")
df.attrs["log"].append(
    "[clean] Values outside plausible ranges set to NaN (including -999 placeholders):\n"
    + pd.DataFrame(
        {
            "outside range": [
                ((df["PM25"] < 0) | (df["PM25"] > 1000)).sum(),
                ((df["PM10"] < 0) | (df["PM10"] > 1000)).sum(),
                ((df["NO2"] < 0) | (df["NO2"] > 500)).sum(),
                ((df["Temperature_C"] < -10) | (df["Temperature_C"] > 55)).sum(),
                ((df["Humidity_Percent"] < 0) | (df["Humidity_Percent"] > 100)).sum(),
            ],
            "-999 placeholders": [
                (df["PM25"] == -999).sum(),
                (df["PM10"] == -999).sum(),
                (df["NO2"] == -999).sum(),
                (df["Temperature_C"] == -999).sum(),
                (df["Humidity_Percent"] == -999).sum(),
            ],
        },
        index=["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"],
    ).to_string()
)
df[["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]] = df[
    ["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]
].mask(
    df[["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]].lt(
        pd.Series(
            [0, 0, 0, -10, 0],
            index=["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"],
        ),
        axis="columns",
    )
    | df[["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"]].gt(
        pd.Series(
            [1000, 1000, 500, 55, 100],
            index=["PM25", "PM10", "NO2", "Temperature_C", "Humidity_Percent"],
        ),
        axis="columns",
    ),
    np.nan,
)

# Keep future dates, but make them visible to the analyst.
df["Future_Date"] = df["Date"].gt(pd.Timestamp.today().normalize())
df.attrs["log"].append(
    f"[clean] {int(df['Future_Date'].sum())} rows have a Date after "
    f"{pd.Timestamp.today().normalize().date()} (flagged in Future_Date, kept)"
)

# PM2.5 imputation: City-Month median with a City-median fallback.
df["PM25_Imputed"] = df["PM25"].isna()
df["PM25"] = df["PM25"].fillna(
    df.groupby(["City", "Month"])["PM25"].transform("median")
).fillna(df.groupby("City")["PM25"].transform("median"))
df.attrs["log"].append(
    f"[51] Imputed {int(df['PM25_Imputed'].sum())} missing/invalid PM2.5 values with "
    f"City-Month median (City-median fallback). Remaining missing PM2.5: {int(df['PM25'].isna().sum())}"
)

# Final checks and outputs.
df = df.sort_values("Date")
df = df.reset_index(drop=True)
df.attrs["log"].extend(
    [
        "[final] Missing values after cleaning:\n" + df.isna().sum().to_string(),
        f"[final] Clean shape: {df.shape}",
    ]
)
df.to_csv("air_quality_clean.csv", index=False)
np.savetxt("cleaning_log.txt", np.array(["\n\n".join(df.attrs["log"])]), fmt="%s")
print("\nSaved air_quality_clean.csv and cleaning_log.txt")

