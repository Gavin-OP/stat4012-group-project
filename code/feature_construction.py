import pandas as pd
from normalization import *

# import PCA feature vectors
pca_data_corr = pd.read_csv('../data/normalized_data_corr_PCA_componants.csv', header=0, index_col=0)
pca_normalized_data = pd.read_csv('../data/normalized_data_PCA_componants.csv', header=0, index_col=0)
pca_diff_corr = pd.read_csv('../data/diff_corr_PCA_componants.csv', header=0, index_col=0)

print('pca_data_corr:\n', pca_data_corr, '\n')
print('pca_normalized_data:\n', pca_normalized_data, '\n')
print('pca_diff_corr:\n', pca_diff_corr, '\n')