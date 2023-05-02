# Description:
# Reshape data into 3D matrix for CNN input
# Dimension of X: (1353 - n_days, n_days, n_features)
# Dimension of y: (1353 - n_days, )

from numpy import array
from feature_construction import PCA_feature_construction
import numpy as np


def input_reshape(n_days=5, stride=1, diff=False):
    feature, label = PCA_feature_construction(diff)
    X = list()
    for i in range(0, len(feature), stride):
        # find the last day of each tensor
        end_index = i + n_days
        # check if we are beyond the dataset
        if end_index > len(feature):
            break

        # reshape input
        # select i to end_index-1 rows and all columns
        seq_X = feature.iloc[i:end_index, :]
        X.append(seq_X)

        # display first 5 samples in a pandas dataframe for better visualization
        # if i < 5:
        #     print('X=%s' % (seq_X))

    y = label.dropna()
    # drop first n_days-1 samples because we don't have X for them
    y = y[n_days-2:]
    # make y with stride = stride
    y = y[::stride].values

    X = array(X)
    y = array(y)
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    return X, y


# below used to test this script
if __name__ == '__main__':
    # X, y = PCA_feature_construction(diff=False)
    X,y = input_reshape(diff=True)
    print(X.shape)

    y = y[~np.isnan(y)]
    print(y.shape)
