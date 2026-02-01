import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fame = pd.read_csv('celebrity_wikipedia_stats.csv')
processed_data = pd.read_csv('2026_MCM_Problem_Labeled_Data.csv')

processed_data['fame_1'] = fame['total_edits']
processed_data['fame_1.1'] = fame['total_edits']**1.1
processed_data['fame_1.2'] = fame['total_edits']**1.2
processed_data['fame_1.3'] = fame['total_edits']**1.3
processed_data['fame_1.4'] = fame['total_edits']**1.4

processed_data.to_csv('2026_MCM_Problem_Fame_Data.csv', index=False)

