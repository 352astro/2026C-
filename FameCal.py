import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

fame = pd.read_csv('combined_season_stats.csv')
processed_data = pd.read_csv('2026_MCM_Problem_Labeled_Data.csv')

processed_data['celebrity_fame_1'] = fame['celebrity_total_edits']
processed_data['celebrity_fame_1.1'] = fame['celebrity_total_edits']**1.1
processed_data['celebrity_fame_1.2'] = fame['celebrity_total_edits']**1.2
processed_data['celebrity_fame_1.3'] = fame['celebrity_total_edits']**1.3
processed_data['celebrity_fame_1.4'] = fame['celebrity_total_edits']**1.4

processed_data['ballroom_fame_1'] = fame['ballroom_total_edits']
processed_data['ballroom_fame_1.1'] = fame['ballroom_total_edits']**1.1
processed_data['ballroom_fame_1.2'] = fame['ballroom_total_edits']**1.2
processed_data['ballroom_fame_1.3'] = fame['ballroom_total_edits']**1.3
processed_data['ballroom_fame_1.4'] = fame['ballroom_total_edits']**1.4

processed_data.to_csv('2026_MCM_Problem_Fame_withBallroom_Data.csv', index=False)

