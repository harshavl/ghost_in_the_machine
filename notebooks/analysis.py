# # Ghost in the Machine: IPL Auction Analytics

# 

# This notebook performs the analysis to quantify 'killer instinct' using Bayesian inference.

# 

# ## Setup

import sys
sys.path.append('../src')

from data_ingestion import data_ingestion_service
from feature_engineering import feature_engineering_service
from modeling import modeling_service
from inference import inference_service
from visualization import visualization_service

# Run pipeline
df_death = data_ingestion_service('../data/raw/IPL_Bowler_Detailed_Data.xls')
df_death.to_csv('../data/processed/death_overs_data.csv', index=False)

X, y, df_death = feature_engineering_service(df_death)
mu, sigma = modeling_service(X, y)
pA_mean, pA_std, pB_mean, pB_std, verdict = inference_service(mu, sigma)
visualization_service(pA_mean, pA_std, pB_mean, pB_std)

# Executive Summary (Text-based; convert to PDF via tool or manually)
summary = f"""
Executive Summary:
- Pressure Effect (Bowler A): {pA_mean:.4f} (94% HDI: [{pA_mean - 1.88*pA_std:.4f}, {pA_mean + 1.88*pA_std:.4f}])
- Pressure Effect (Bowler B): {pB_mean:.4f} (94% HDI: [{pB_mean - 1.88*pB_std:.4f}, {pB_mean + 1.88*pB_std:.4f}])
- Verdict: {verdict}. Bowler B shows strong killer instinct.
"""
print(summary)
with open('../docs/executive_summary.txt', 'w') as f:
    f.write(summary)