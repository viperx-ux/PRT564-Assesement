"""
NSW Road Crash Data 2020-2024
Assessment 2 - Data Preprocessing & Exploratory Data Analysis
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


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 11
})
colors = ['#5DCAA5', '#7F77DD', '#D85A30', '#378ADD', '#BA7517']

# ─── Plot 1: Crash Severity Distribution ─────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Figure 1: Crash Severity Distribution', fontsize=13, fontweight='bold', y=1.01)

# Bar chart
crash_counts = df['Degree of crash'].value_counts()
axes[0].bar(crash_counts.index, crash_counts.values, color=colors[:3], edgecolor='white', linewidth=0.5)
axes[0].set_title('Count by severity level')
axes[0].set_ylabel('Number of crashes')
for i, v in enumerate(crash_counts.values):
    axes[0].text(i, v + 300, f'{v:,}', ha='center', fontsize=10)

# Pie chart
axes[1].pie(crash_counts.values, labels=crash_counts.index, autopct='%1.1f%%',
            colors=colors[:3], startangle=90, textprops={'fontsize': 10})
axes[1].set_title('Proportion by severity level')

plt.tight_layout()
plt.savefig('fig1_severity_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig1_severity_distribution.png")

# ─── Plot 2: Crashes by Year and Month ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 2: Temporal Patterns in Road Crashes', fontsize=13, fontweight='bold')

year_counts = df.groupby('Year of crash').size()
axes[0].plot(year_counts.index, year_counts.values, marker='o', color=colors[0], linewidth=2.5, markersize=7)
axes[0].set_title('Total crashes per year')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Number of crashes')
axes[0].set_xticks(year_counts.index)
for x, y in zip(year_counts.index, year_counts.values):
    axes[0].annotate(f'{y:,}', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

month_counts = df.groupby('Month_num').size()
month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
axes[1].bar(month_counts.index, month_counts.values, color=colors[1], edgecolor='white')
axes[1].set_title('Total crashes by month')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Number of crashes')
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(month_labels, rotation=45)

plt.tight_layout()
plt.savefig('fig2_temporal_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig2_temporal_patterns.png")

# ─── Plot 3: Environmental Factors ───────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Figure 3: Environmental Conditions at Time of Crash', fontsize=13, fontweight='bold')

weather_counts = df['Weather'].value_counts()
axes[0].barh(weather_counts.index, weather_counts.values, color=colors[3])
axes[0].set_title('Weather condition')
axes[0].set_xlabel('Number of crashes')

light_counts = df['Natural lighting'].value_counts()
axes[1].barh(light_counts.index, light_counts.values, color=colors[0])
axes[1].set_title('Natural lighting')
axes[1].set_xlabel('Number of crashes')

surface_counts = df['Surface condition'].value_counts().head(6)
axes[2].barh(surface_counts.index, surface_counts.values, color=colors[2])
axes[2].set_title('Road surface condition')
axes[2].set_xlabel('Number of crashes')

plt.tight_layout()
plt.savefig('fig3_environmental_factors.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig3_environmental_factors.png")

# ─── Plot 4: Speed Limit vs Severity ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 4: Speed Limit vs Crash Severity', fontsize=13, fontweight='bold')

speed_severity = df.groupby(['Speed limit', 'Degree of crash']).size().unstack(fill_value=0)
speed_order = ['10 km/h','20 km/h','30 km/h','40 km/h','50 km/h','60 km/h',
               '70 km/h','80 km/h','90 km/h','100 km/h','110 km/h']
speed_severity = speed_severity.reindex([s for s in speed_order if s in speed_severity.index])
speed_severity.plot(kind='bar', ax=axes[0], color=colors[:3], edgecolor='white')
axes[0].set_title('Crash count by speed zone')
axes[0].set_xlabel('Speed limit')
axes[0].set_ylabel('Number of crashes')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Severity')

# Fatal rate by speed limit
fatal_rate = df.groupby('Speed limit').apply(
    lambda x: (x['Degree of crash'] == 'Fatal').sum() / len(x) * 100
).reindex([s for s in speed_order if s in df['Speed limit'].unique()])
axes[1].bar(range(len(fatal_rate)), fatal_rate.values, color=colors[2], edgecolor='white')
axes[1].set_xticks(range(len(fatal_rate)))
axes[1].set_xticklabels(fatal_rate.index, rotation=45)
axes[1].set_title('Fatal crash rate (%) by speed zone')
axes[1].set_ylabel('Fatal crash rate (%)')

plt.tight_layout()
plt.savefig('fig4_speed_severity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig4_speed_severity.png")

# ─── Plot 5: Urbanisation & Day of Week ──────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 5: Location and Day-of-Week Patterns', fontsize=13, fontweight='bold')

urban_fatal = df.groupby('Urbanisation').apply(
    lambda x: pd.Series({
        'Total crashes': len(x),
        'Fatal crashes': (x['Degree of crash'] == 'Fatal').sum()
    })
)
urban_fatal['Fatal rate %'] = (urban_fatal['Fatal crashes'] / urban_fatal['Total crashes'] * 100).round(2)
urban_fatal = urban_fatal.sort_values('Fatal rate %', ascending=True)
axes[0].barh(urban_fatal.index, urban_fatal['Fatal rate %'], color=colors[1])
axes[0].set_title('Fatal crash rate by urbanisation')
axes[0].set_xlabel('Fatal crash rate (%)')

day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
day_severity = df.groupby(['Day of week of crash','Degree of crash']).size().unstack(fill_value=0)
day_severity = day_severity.reindex(day_order)
day_severity.plot(kind='bar', ax=axes[1], color=colors[:3], edgecolor='white')
axes[1].set_title('Crashes by day of week and severity')
axes[1].set_xlabel('Day')
axes[1].set_ylabel('Number of crashes')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Severity', fontsize=9)

plt.tight_layout()
plt.savefig('fig5_location_day_patterns.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig5_location_day_patterns.png")

# ─── Plot 6: Correlation Heatmap ─────────────
numeric_cols = [
    'Speed_limit_num', 'No. of traffic units involved',
    'No. killed', 'No. seriously injured', 'No. moderately injured',
    'No. minor-other injured', 'Total_casualties', 'Severity_score',
    'Is_dark', 'Is_wet', 'Is_weekend', 'Crash_severity_encoded'
]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 9))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
    center=0, ax=ax, linewidths=0.5, annot_kws={'size': 9},
    cbar_kws={'shrink': 0.8}
)
ax.set_title('Figure 6: Correlation Heatmap of Numeric Features', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('fig6_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig6_correlation_heatmap.png")

# ─── Plot 7: Severity Score Distribution ─────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Figure 7: Casualty & Severity Score Distributions', fontsize=13, fontweight='bold')

# Only crashes that had casualties
with_casualties = df[df['Total_casualties'] > 0]
axes[0].hist(with_casualties['Total_casualties'].clip(upper=10), bins=10,
             color=colors[0], edgecolor='white')
axes[0].set_title('Total casualties per crash (capped at 10)')
axes[0].set_xlabel('Total casualties')
axes[0].set_ylabel('Frequency')

axes[1].hist(with_casualties['Severity_score'].clip(upper=15), bins=15,
             color=colors[4], edgecolor='white')
axes[1].set_title('Weighted severity score per crash (capped at 15)')
axes[1].set_xlabel('Severity score')
axes[1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('fig7_severity_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig7_severity_distributions.png")

# ─── Plot 8: Top Locations ─────────────
top_lga = df['LGA'].value_counts().head(10)

plt.figure(figsize=(8, 5))
top_lga.plot(kind='barh', color='#378ADD')
plt.title('Top 10 LGAs by Crash Frequency')
plt.xlabel('Number of Crashes')
plt.ylabel('LGA')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('fig6_top_lga.png', dpi=150)
plt.close()

print("Saved: fig6_top_lga.png")

# ─────────────────────────────────────────────
# 3. STATISTICAL SUMMARY FOR PRESENTATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("KEY STATISTICAL FINDINGS")
print("=" * 60)

print(f"\nTotal crashes (2020-2024):    {len(df):,}")
print(f"Fatal crashes:                {df['Is_fatal'].sum():,} ({df['Is_fatal'].mean()*100:.2f}%)")
print(f"Total casualties:             {df['Total_casualties'].sum():,}")
print(f"Total fatalities:             {df['No. killed'].sum():,}")

print(f"\nCrashes in darkness/dusk/dawn: {df['Is_dark'].sum():,} ({df['Is_dark'].mean()*100:.1f}%)")
print(f"Crashes on wet roads:          {df['Is_wet'].sum():,} ({df['Is_wet'].mean()*100:.1f}%)")
print(f"Weekend crashes:               {df['Is_weekend'].sum():,} ({df['Is_weekend'].mean()*100:.1f}%)")

# Chi-square: darkness vs fatal
ct = pd.crosstab(df['Is_dark'], df['Is_fatal'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
print(f"\nChi-square test (darkness vs fatal crash):")
print(f"  chi2={chi2:.2f}, p={p:.4f}, dof={dof}")
print(f"  → {'Significant' if p < 0.05 else 'Not significant'} at alpha=0.05")

# Chi-square: wet road vs fatal
ct2 = pd.crosstab(df['Is_wet'], df['Is_fatal'])
chi2_2, p2, dof2, _ = stats.chi2_contingency(ct2)
print(f"\nChi-square test (wet road vs fatal crash):")
print(f"  chi2={chi2_2:.2f}, p={p2:.4f}, dof={dof2}")
print(f"  → {'Significant' if p2 < 0.05 else 'Not significant'} at alpha=0.05")

# Speed correlation with severity
spearman_corr, spearman_p = stats.spearmanr(
    df['Speed_limit_num'].dropna(),
    df.loc[df['Speed_limit_num'].notna(), 'Crash_severity_encoded']
)
print(f"\nSpearman correlation (speed limit vs crash severity):")
print(f"  rho={spearman_corr:.3f}, p={spearman_p:.4f}")
print(f"  → {'Significant positive' if spearman_p < 0.05 and spearman_corr > 0 else 'Not significant'} relationship")

# Save cleaned dataset
df.to_csv('nsw_crash_preprocessed.csv', index=False)
print(f"\nPreprocessed dataset saved: nsw_crash_preprocessed.csv")
print(f"Final shape: {df.shape}")
print("\nAll preprocessing and EDA steps complete.")
