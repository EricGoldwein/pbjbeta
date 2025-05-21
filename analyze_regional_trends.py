import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
region_metrics = pd.read_csv('region_quarterly_metrics.csv')

# Convert CY_Qtr to datetime for better sorting
region_metrics['date'] = pd.to_datetime(region_metrics['CY_Qtr'].str[:4] + 'Q' + region_metrics['CY_Qtr'].str[-1])
region_metrics = region_metrics.sort_values(['Region', 'date'])

# Metrics to analyze
metrics_to_analyze = ['Total_HPRD', 'RN_HPRD', 'Nurse_Assistant_HPRD', 'Contract_Staff_Percentage']

# Create visualization
plt.figure(figsize=(20, 15))

for i, metric in enumerate(metrics_to_analyze, 1):
    plt.subplot(2, 2, i)
    
    # Plot each region's trend
    for region in sorted(region_metrics['Region'].unique()):
        region_data = region_metrics[region_metrics['Region'] == region]
        plt.plot(region_data['date'], region_data[metric], 
                marker='o', markersize=4, 
                label=region, 
                alpha=0.7,
                linewidth=2)
    
    plt.title(f'{metric} Trends by Region', fontsize=14, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('regional_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# Create boxplots to show distribution
plt.figure(figsize=(20, 15))

for i, metric in enumerate(metrics_to_analyze, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(data=region_metrics, x='Region', y=metric)
    plt.title(f'{metric} Distribution by Region', fontsize=14, pad=20)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regional_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nVisualization complete. Check 'regional_trends.png' and 'regional_distributions.png' for the visual analysis.") 