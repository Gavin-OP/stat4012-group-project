# Description:
# Create new features according to PCA analysis

import pandas as pd
import matplotlib.pyplot as plt
from normalization import *

# import PCA feature vectors
pca_normalized_data_diff = pd.read_csv(
    '../data/normalized_data_diff_PCA_componants.csv', header=0, index_col=0)
pca_normalized_data = pd.read_csv(
    '../data/normalized_data_PCA_componants.csv', header=0, index_col=0)

print('pca_normalized_data_diff:\n', pca_normalized_data_diff, '\n')
print('pca_normalized_data:\n', pca_normalized_data, '\n')

pca_componants = pca_normalized_data # use pca_normalized_data_diff or pca_normalized_data
# create new features using PCA
golden_cross = data['golden_cross']
death_cross = data['death_cross']
sixth_day_return = data['sixth_day_return']
drop_columns = ['golden_cross', 'death_cross', 'sixth_day_return']
data_temp = data.drop(drop_columns, axis=1, inplace=True)
X = pd.DataFrame(index=data.index)

for i in pca_componants.columns:
    # make new features by PCA componants matrix multiply with the original features
    X[i] = data @ pca_componants[i]

# add golden cross, death cross to X
X['golden_cross'] = golden_cross
X['death_cross'] = death_cross

print(X.head())
y = sixth_day_return
print(y.head())

# plot the new features only display 5 x-axis labels
plt.figure(figsize=(10, 8))
plt.plot(X['Comp.1'], label='Comp.1')
plt.plot(X['Comp.2'], label='Comp.2')
plt.plot(X['Comp.3'], label='Comp.3')
plt.plot(X['Comp.4'], label='Comp.4')
plt.plot(X['Comp.5'], label='Comp.5')
plt.scatter(X[X['golden_cross'] == 1].index,
            X[X['golden_cross'] == 1]['Comp.2'], label='golden_cross', marker='^', color='green')
plt.scatter(X[X['death_cross'] == 1].index,
            X[X['death_cross'] == 1]['Comp.2'], label='golden_cross', marker='v', color='red')
plt.xticks(X.index[::int(len(X) / 5)])
plt.legend()
plt.show()

# plot label only display 5 x-axis labels
plt.figure(figsize=(10, 8))
plt.plot(y, label='sixth_day_return')
plt.xticks(y.index[::int(len(y) / 5)])
plt.legend()
plt.show()
