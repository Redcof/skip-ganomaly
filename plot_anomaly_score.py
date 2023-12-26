import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

score_df = pd.read_csv("output/sixray_sd3_grey_exp_px64/anomaly_score-epoch-10.csv")

plt.ion()
# Filter normal and abnormal scores.
abn_scr = score_df.loc[score_df.labels == 1]['scores']
nrm_scr = score_df.loc[score_df.labels == 0]['scores']

# Create figure and plot the distribution.
sns.distplot(nrm_scr, label=r'Normal Scores', bins=10)
sns.distplot(abn_scr, label=r'Abnormal Scores', bins=10)

plt.legend()
plt.yticks([])
plt.xlabel(r'Anomaly Scores')
plt.show(block=True)
