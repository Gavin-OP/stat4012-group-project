# Description:
# Reshape data into 2D matrix for CNN input
# Split data into train and test set
# Dimension of X_train, X_test, y_train, y_test:
#

from numpy import array
from feature_construction import *
# Define reshape function to reshape the data into matrix for CNN input.
# split a 1322 * 21 matrix into (1322-n_days+1)/stride * n_days * 20 tensorflow matrix


def feature_reshape(feature, n_days=5, stride=1):
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
        if i < 5:
            print('X=%s' % (seq_X))
        

    return array(X)


print(feature_reshape(X))
