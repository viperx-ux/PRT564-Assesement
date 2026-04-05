"""
NSW Road Crash Data 2020-2024
Assessment 2 - Data Preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 0. Load Data
# ─────────────────────────────────────────────
df = pd.read_excel('nsw_road_crash_data_2020-2024_crash.xlsx')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")

# ─────────────────────────────────────────────
# 1. DATA PREPROCESSING
# ─────────────────────────────────────────────

print("=" * 60)
print("STEP 1: DATA PREPROCESSING")
print("=" * 60)

# --- 1a. Drop index column (not a feature) ---
df.drop(columns=['j'], inplace=True)

# --- 1b. Missing Value Analysis ---
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False)
print("\nMissing Values Summary:")
print(missing_df.to_string())

# --- 1c. Handle Missing Values ---

# Columns with >70% missing → drop (not informative enough)
high_missing_cols = missing_df[missing_df['Missing %'] > 70].index.tolist()
print(f"\nDropping columns with >70% missing: {high_missing_cols}")
df.drop(columns=high_missing_cols, inplace=True)

# 'Route no.' (~37% missing) → fill with 0 (no route number assigned)
df['Route no.'] = df['Route no.'].fillna(0)

# 'Other TU type' (~30% missing) → fill with 'None'
df['Other TU type'] = df['Other TU type'].fillna('None')

# 'DCA supplement' (~87% missing) → already dropped above

# 'Identifying feature' (1 row missing) → fill with 'Unknown'
df['Identifying feature'] = df['Identifying feature'].fillna('Unknown')

print(f"\nRemaining columns after cleanup: {df.columns.tolist()}")
print(f"\nRemaining missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# --- 1d. Fix 'Unknown' categories in categorical columns ---
# Replace 'Unknown' with NaN, then fill with mode for low-% unknowns
low_unknown_cols = ['Weather', 'Natural lighting', 'Road surface', 'Speed limit']
for col in low_unknown_cols:
    df[col] = df[col].replace('Unknown', np.nan)
    mode_val = df[col].mode()[0]
    df[col] = df[col].fillna(mode_val)
    print(f"  '{col}' Unknowns filled with mode: '{mode_val}'")

# --- 1e. Create engineered features ---

# Total casualties per crash
df['Total_casualties'] = (
    df['No. killed'] +
    df['No. seriously injured'] +
    df['No. moderately injured'] +
    df['No. minor-other injured']
)

# Severity score (weighted: fatal=4, serious=3, moderate=2, minor=1)
df['Severity_score'] = (
    df['No. killed'] * 4 +
    df['No. seriously injured'] * 3 +
    df['No. moderately injured'] * 2 +
    df['No. minor-other injured'] * 1
)

# Month as numeric (for time-series ordering)
month_map = {'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
             'July':7,'August':8,'September':9,'October':10,'November':11,'December':12}
df['Month_num'] = df['Month of crash'].map(month_map)

# Day of week as numeric
day_map = {'Monday':1,'Tuesday':2,'Wednesday':3,'Thursday':4,'Friday':5,'Saturday':6,'Sunday':7}
df['Day_num'] = df['Day of week of crash'].map(day_map)

# Speed limit as numeric (extract number from string like '50 km/h')
df['Speed_limit_num'] = df['Speed limit'].str.extract(r'(\d+)').astype(float)

# Binary: is_fatal
df['Is_fatal'] = (df['Degree of crash'] == 'Fatal').astype(int)

# Binary: is_weekend
df['Is_weekend'] = df['Day of week of crash'].isin(['Saturday', 'Sunday']).astype(int)

# Binary: is_dark
df['Is_dark'] = df['Natural lighting'].isin(['Darkness', 'Dusk', 'Dawn']).astype(int)

# Binary: is_wet
df['Is_wet'] = df['Surface condition'].str.lower().str.contains('wet|damp', na=False).astype(int)

print("\nFeature engineering complete. New columns added:")
new_cols = ['Total_casualties', 'Severity_score', 'Month_num', 'Day_num',
            'Speed_limit_num', 'Is_fatal', 'Is_weekend', 'Is_dark', 'Is_wet']
print(new_cols)

# --- 1f. Encode ordinal target for regression ---
# Degree of crash → numeric severity (0=Non-casualty, 1=Injury, 2=Fatal)
crash_severity_map = {'Non-casualty (towaway)': 0, 'Injury': 1, 'Fatal': 2}
df['Crash_severity_encoded'] = df['Degree of crash'].map(crash_severity_map)

print(f"\nFinal preprocessed dataset shape: {df.shape}")
print(f"\nSample of new features:\n{df[new_cols + ['Crash_severity_encoded']].describe().round(2)}")

