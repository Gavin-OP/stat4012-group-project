# Description:
# Create new features according to PCA analysis

import pandas as pd
import matplotlib.pyplot as plt
from normalization import normalize_data

# import PCA feature vectors
def import_PCA_componants(diff=False):
    pca_normalized_data_diff = pd.read_csv(
        '../data/normalized_data_diff_PCA_componants.csv', header=0, index_col=0)
    pca_normalized_data = pd.read_csv(
        '../data/normalized_data_PCA_componants.csv', header=0, index_col=0)
    # print('pca_normalized_data_diff:\n', pca_normalized_data_diff, '\n')
    # print('pca_normalized_data:\n', pca_normalized_data, '\n')
    pca_componants = pd.DataFrame()
    if diff == True:
        pca_componants = pca_normalized_data_diff
    elif diff == False:
        pca_componants = pca_normalized_data
    return pca_componants


# create new features using PCA, diff = True means using normalized_data_diff_PCA_componants.csv, diff = False means using normalized_data_PCA_componants.csv
def PCA_feature_construction(diff = False):
    pca_componants = import_PCA_componants(diff)
    data = normalize_data()
    golden_cross = data['golden_cross']
    death_cross = data['death_cross']
    label = data['return']
    drop_columns = ['golden_cross', 'death_cross', 'return']
    data.drop(drop_columns, axis=1, inplace=True)
    X = pd.DataFrame(index=data.index)

    for i in pca_componants.columns:
        # make new features by PCA componants matrix multiply with the original features
        X[i] = data @ pca_componants[i]

    # add golden cross, death cross to X
    X['golden_cross'] = golden_cross
    X['death_cross'] = death_cross

    # use sixth_day_return as y
    y = label
    return X, y

# plot the new features only display 5 x-axis labels
def new_features_plot(X):
    n = len(X.columns)
    print('n:', n)
    plt.figure(figsize=(10, 8))

    plt.plot(X['Comp.1'], label='Comp.1')
    plt.plot(X['Comp.2'], label='Comp.2')
    plt.plot(X['Comp.3'], label='Comp.3')
    # plot golden cross and death cross along x axis
    plt.scatter(X[X['golden_cross'] == 1].index,
                X[X['golden_cross'] == 1]['Comp.1'] * 0, label='golden_cross', marker='^', color='green')
    plt.scatter(X[X['death_cross'] == 1].index,
                X[X['death_cross'] == 1]['Comp.1'] * 0, label='golden_cross', marker='v', color='red')
    # plt.scatter(X[X['golden_cross'] == 1].index,
    #             X[X['golden_cross'] == 1]['Comp.2'], label='golden_cross', marker='^', color='green')
    # plt.scatter(X[X['death_cross'] == 1].index,
    #             X[X['death_cross'] == 1]['Comp.2'], label='golden_cross', marker='v', color='red')
    plt.xticks(X.index[::int(len(X) / 5)])
    plt.legend()
    plt.show()

# plot label only display 5 x-axis labels
def label_plot(y):
    plt.figure(figsize=(10, 8))
    plt.plot(y, label='return')
    plt.xticks(y.index[::int(len(y) / 5)])
    plt.legend()
    plt.show()

if __name__ ==  '__main__':
    # usage of function defined above
    X, y = PCA_feature_construction(diff=False)
    X1, y1 = PCA_feature_construction(diff=True)
    print('X:\n', X, '\n')
    print('y:\n', y, '\n')
    new_features_plot(X)
    new_features_plot(X1)
    label_plot(y)
