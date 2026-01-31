import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('2026_MCM_Problem_Mean_Data.csv')
array = data.sort_values(by = 'placement')
plt.figure(figsize=(10, 6))

# Assume data is the DataFrame from CSV
# data = pd.read_csv('2026_MCM_Problem_Mean_Data.csv')

plt.figure(figsize=(10, 6))
plt.scatter(data['1'], data['weeks'], alpha=0.6, color='blue')
plt.gca().invert_yaxis()  # Invert y to have 1 (best) at top
plt.title('Week 1 Judge Average vs Placement')
plt.xlabel('Week 1 Judge Average Score')
plt.ylabel('Placement (1 = Best)')
plt.grid(True)
plt.show()