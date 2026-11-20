
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/holdout_features.csv")

print("TRAIN SHAPE:", train.shape)
print("LATER SHAPE:", later.shape)
print("HOLDOUT SHAPE:", holdout.shape)


# ==========================================
# 2. DUPLICATE ROWS
# ==========================================

print("\n1. DUPLICATE ROWS")

duplicates = train.duplicated().sum()
print("Duplicate rows:", duplicates)


# ==========================================
# 3. UNIT MIX-UP
# ==========================================

print("\n2. UNIT MIX-UP: HIGHWAY DISTANCE")

unit_check = train.groupby("zone")["highway_distance_km"].agg(
    ["min", "mean", "median", "max"]
)

print(unit_check)

industrial_mean = unit_check.loc["Industrial", "mean"]
normal_mean = unit_check.drop(index="Industrial")["mean"].median()

print("Industrial / Normal ratio:", industrial_mean / normal_mean)


# ==========================================
# 4. SENTINEL VALUES
# ==========================================

print("\n3. SENTINEL VALUES")

sentinels = [-9999, -999, -1, 999, 9999, 99999]

sentinel_found = []

for col in train.select_dtypes(include="number").columns:

    for value in sentinels:

        count = (train[col] == value).sum()

        if count > 0:
            sentinel_found.append((col, value, count))
            print(col, "Value:", value, "Count:", count)

if not sentinel_found:
    print("No sentinel values detected.")


# ==========================================
# 5. STUCK SENSOR
# ==========================================

print("\n4. STUCK SENSOR: TEMPERATURE")

temp_data = train[["date", "temp_c"]].copy()

temp_data["date"] = pd.to_datetime(temp_data["date"])

daily_temp = (
    temp_data.groupby("date")["temp_c"]
    .agg(["min", "max"])
    .sort_index()
)

# Dates where temperature remained constant
stuck_dates = daily_temp[
    daily_temp["min"] == daily_temp["max"]
]

print("Constant temperature dates:")
print(stuck_dates)


# Find longest consecutive run with the same temperature
runs = []

if not stuck_dates.empty:

    current_dates = []
    previous_date = None
    previous_temp = None

    for date, row in stuck_dates.iterrows():

        value = row["min"]

        if (
            previous_date is not None
            and (date - previous_date).days == 1
            and value == previous_temp
        ):
            current_dates.append(date)

        else:
            if current_dates:
                runs.append(current_dates)

            current_dates = [date]

        previous_date = date
        previous_temp = value

    if current_dates:
        runs.append(current_dates)

if runs:
    longest_run = max(runs, key=len)

    print("Longest stuck sensor run:")
    print("Start:", longest_run[0].strftime("%Y-%m-%d"))
    print("End:", longest_run[-1].strftime("%Y-%m-%d"))
    print("Number of days:", len(longest_run))

else:
    print("No consecutive stuck sensor run detected.")


# ==========================================
# 6. TARGET TYPO CHECK
# ==========================================

print("\n5. TARGET TYPO CHECK")

targets = np.sort(train["litres_per_day"].dropna().unique())

gaps = np.diff(targets)

print("Largest target gaps:")

for i in np.argsort(gaps)[-10:][::-1]:

    print(
        targets[i],
        "to",
        targets[i + 1],
        "Gap:",
        gaps[i]
    )

# Check whether unusually large targets are 10 times
# another existing target value

target_values = train["litres_per_day"]

normal_targets = targets[targets > 0]

typo_candidates = train[
    target_values.notna()
    & (target_values > 0)
    & target_values.apply(
        lambda x: np.any(
            np.isclose(normal_targets, x / 10, rtol=0.001)
        )
    )
]

print("\nPossible extra-zero target rows:")
print(
    typo_candidates[
        ["record_id", "litres_per_day"]
    ].to_string(index=False)
)

print("Possible target typo count:", len(typo_candidates))


# ==========================================
# 7. SPELLING VARIANTS
# ==========================================

print("\n6. SPELLING VARIANTS")

for col in train.select_dtypes(include="object").columns:

    print("\nColumn:", col)

    print("Train values:")
    print(train[col].value_counts(dropna=False))

    if col in later.columns:

        print("Later values:")
        print(later[col].value_counts(dropna=False))

        train_clean = set(
            train[col].dropna().astype(str).str.strip().str.lower()
        )

        later_clean = set(
            later[col].dropna().astype(str).str.strip().str.lower()
        )

        print("Clean train categories:", train_clean)
        print("Clean later categories:", later_clean)


# ==========================================
# 8. POSSIBLE DATA LEAKAGE
# ==========================================

print("\n7. POSSIBLE DATA LEAKAGE")

missing_from_holdout = [
    col for col in train.columns
    if col not in holdout.columns
]

print("Train columns missing in holdout:", missing_from_holdout)

# Target is expected to be absent from holdout.
# Do not automatically call it leakage.

feature_missing = [
    col for col in missing_from_holdout
    if col != "litres_per_day"
]

print("Missing feature columns:", feature_missing)

if feature_missing:
    print("Investigate these columns for leakage.")
else:
    print("No additional missing feature column detected.")


# ==========================================
# 9. TARGET CORRELATIONS
# ==========================================

print("\nTARGET CORRELATIONS")

print(
    train.select_dtypes(include="number")
    .corr()["litres_per_day"]
    .sort_values(ascending=False)
)


# ==========================================
# 10. FINAL VERIFICATION
# ==========================================

print("\n--- FINAL VERIFICATION ---")

print("\nOwnership values:")

print("Train:")
print(train["ownership"].value_counts())

print("\nLater:")
print(later["ownership"].value_counts())

print("\nTrain columns:")
print(train.columns.tolist())

print("\nHoldout columns:")
print(holdout.columns.tolist())

print("\nANALYSIS COMPLETE")
