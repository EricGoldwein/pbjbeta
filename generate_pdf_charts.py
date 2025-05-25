import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('C:/Users/egold/PycharmProjects/NewPBJ/national_pbj_metrics.csv')

# Create year labels for x-axis ticks
min_year = int(df['Quarter'].str[:4].min())
max_year = int(df['Quarter'].str[:4].max())
all_years = range(min_year, max_year + 1)
tick_values = [f"{year}Q1" for year in all_years]
tick_text = [str(year) for year in all_years]

# 1. US Total Staffing Levels (2017-2024)
plt.figure(figsize=(10, 6))
plt.plot(df['Quarter'], df['Total_Nurse_HPRD'], marker='o')
plt.title('US Total Staffing Levels (2017-2024)')
plt.xlabel('Quarter')
plt.ylabel('Hours Per Resident Day - HPRD')
plt.xticks(tick_values, tick_text, rotation=45)
plt.figtext(0.98, 0.01, '320 Consulting | Source: CMS PBJ Data', ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('us_total_staffing_levels.pdf')
plt.close()

# 2. US RN Staffing Levels (2017-2024)
plt.figure(figsize=(10, 6))
plt.plot(df['Quarter'], df['RN_HPRD'], marker='o')
plt.title('US RN Staffing Levels (2017-2024)')
plt.xlabel('Quarter')
plt.ylabel('Hours Per Resident Day - HPRD')
plt.xticks(tick_values, tick_text, rotation=45)
plt.figtext(0.98, 0.01, '320 Consulting | Source: CMS PBJ Data', ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('us_rn_staffing_levels.pdf')
plt.close()

# 3. US Contract Staff % (2017-2024)
plt.figure(figsize=(10, 6))
plt.plot(df['Quarter'], df['Contract_Percentage'], marker='o')
plt.title('US Contract Staff % (2017-2024)')
plt.xlabel('Quarter')
plt.ylabel('Share of contract staff (%)')
plt.xticks(tick_values, tick_text, rotation=45)
plt.figtext(0.98, 0.01, '320 Consulting | Source: CMS PBJ Data', ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('us_contract_staff_percentage.pdf')
plt.close()

# 4. US Nursing Home Residents (2017-2024)
plt.figure(figsize=(10, 6))
plt.plot(df['Quarter'], df['Total_MDScensus'], marker='o')
plt.title('US Nursing Home Residents (2017-2024)')
plt.xlabel('Quarter')
plt.ylabel('MDS census (thousands)')
plt.xticks(tick_values, tick_text, rotation=45)
plt.figtext(0.98, 0.01, '320 Consulting | Source: CMS PBJ Data', ha='right', fontsize=8, color='gray')
plt.tight_layout()
plt.savefig('us_nursing_home_residents.pdf')
plt.close() 