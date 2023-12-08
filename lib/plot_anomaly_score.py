import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

score_df = pd.read_csv("../output/skipganomaly/sixray_sd3/anomaly_score-epoch-0.csv")

plt.ion()
# Filter normal and abnormal scores.
abn_scr = score_df.loc[score_df.labels == 1]['scores']
nrm_scr = score_df.loc[score_df.labels == 0]['scores']

# Create figure and plot the distribution.
sns.distplot(nrm_scr, label=r'Normal Scores')
sns.distplot(abn_scr, label=r'Abnormal Scores')

plt.legend()
plt.yticks([])
plt.xlabel(r'Anomaly Scores')
