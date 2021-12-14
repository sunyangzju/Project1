import numpy as np
import math
import random
import itertools
import pandas as pd
import statistics
from collections import Counter
from numpy import linalg as LA
import seaborn as sns
import scipy
from fancyimpute import SoftImpute, IterativeImputer, BiScaler

softImpute = SoftImpute(max_rank=10)  # simulate data is rank 10!
biscaler = BiScaler()
import matplotlib.pyplot as plt


from missingpy import MissForest
# from missingpy import SY_MI_GCDS
from missingpy import MI_GCDS


from gain import gain
# % matplotlib
# inline
# %% md
## Data simulation
# %%
# generate the rating matrix (full)
# Missing Data Simulation for MCAR and MAR

import numpy as np
import random
from sklearn.impute import SimpleImputer

# y_miss = y_mcar
# y
# LA.norm(y[np.isnan(y_miss)])
# np.isnan(y_miss).shape
# np.sum(np.isnan(y_miss))
def missmean(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1
    # y_softimpute = softImpute.fit_transform(y_s)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    y_original = imp.fit_transform(y_miss)
    print(y_original)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))
    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)]) ) ** 2
    return test_error


def soft(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1
    # y_softimpute = softImpute.fit_transform(y_s)
    y_original = softImpute.fit_transform(y_miss)
    # y_original = biscaler.inverse_transform(y_softimpute)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))
    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)])) ** 2
    return test_error

def soft_mask(y_miss, y):
    # ncol = np.shape(y)[1]
    mask = np.isnan(y_miss).astype(int)
    # y_input_s = biscaler.fit_transform(y_miss)
    y_input = np.concatenate((y_miss, mask), axis=1)
    y_MCAR_original = softImpute.fit_transform(y_input)
    # y_MCAR_original = biscaler.inverse_transform(y_MCAR_softimpute[:, 0:ncol])
    test_error = LA.norm(y[np.isnan(y_miss)] - y_MCAR_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))
    # test_error = (LA.norm(y_original[np.isnan(y_miss)] - y_MCAR_original[np.isnan(y_miss)]) / LA.norm(
    #     y_original[np.isnan(y_miss)])) ** 2
    return test_error
# %%
def MI(y_miss, y):
    n_imputations = 5
    XY_completed = []
    for i in range(n_imputations):
        imputer = IterativeImputer(max_iter=5, sample_posterior=True, random_state=i)
        XY_completed.append(imputer.fit_transform(y_miss))
    completed_mean = np.mean(XY_completed, 0)  # mean of the imputed matrix
    test_error = LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))

    # test_error_MI = (LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / LA.norm(
    #     y[np.isnan(y_miss)])) ** 2
    return test_error

def MIKNN(y_miss, y):
    n_imputations = 5
    XY_completed = []
    for i in range(n_imputations):
        imputer = IterativeImputer(max_iter=5, n_nearest_features=5, sample_posterior=True, random_state=i)
        XY_completed.append(imputer.fit_transform(y_miss))
    completed_mean = np.mean(XY_completed, 0)  # mean of the imputed matrix
    test_error = LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))
    # test_error_MI = (LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / LA.norm(
    #     y[np.isnan(y_miss)])) ** 2
    return test_error

def MF(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1

    imputer = MissForest()
    y_original = imputer.fit_transform(y_miss)

    # y_softimpute = softImpute.fit_transform(y_s)
    # y_original = biscaler.inverse_transform(X_imputed)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))

    # dfcor = pd.DataFrame(y_original)
    # Var_Corr = dfcor.corr()
    # # plot the heatmap and annotation on it
    # sns.heatmap(Var_Corr, vmin=0, vmax=1, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    # plt.text(0.5, 1.1, "Heat Map" , fontsize = 10)
    # plt.show()

    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)])) ** 2
    return test_error


def MI_GCDS_1(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1

    imputer = MI_GCDS()
    y_original = imputer.fit_transform(y_miss)
    print(y_original)

    # y_softimpute = softImpute.fit_transform(y_s)
    # y_original = biscaler.inverse_transform(X_imputed)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))

    # dfcor = pd.DataFrame(y_original)
    # Var_Corr = dfcor.corr()
    # # plot the heatmap and annotation on it
    # sns.heatmap(Var_Corr, vmin=0, vmax=1, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    # plt.text(0.5, 1.1, "Heat Map" , fontsize = 10)
    # plt.show()

    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)])) ** 2
    return test_error


def GAIN(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1

    gain_parameters = {'batch_size': 32,
                       'hint_rate': .9,
                       'alpha': 100,
                       'iterations': 10000}
    # Impute missing data
    y_original = gain(np.array(y_miss), gain_parameters)

    # imputer = MissForest()
    # X_imputed = imputer.fit_transform(y_s)

    # y_softimpute = softImpute.fit_transform(y_s)
    # y_original = biscaler.inverse_transform(imputed_data_x)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))

    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)])) ** 2
    return test_error



def ensemble_MIMF(y_miss, y):
    # y_s = biscaler.fit_transform(y_miss)  # standardized to row/col with mean 0 and var =1

    imputer = MI_GCDS()
    y_original_MF = imputer.fit_transform(y_miss)


    n_imputations = 5
    XY_completed = []
    for i in range(n_imputations):
        imputer = IterativeImputer(max_iter=5, sample_posterior=True, random_state=i)
        XY_completed.append(imputer.fit_transform(y_miss))
    y_original_MI = np.mean(XY_completed, 0)  # mean of the imputed matrix

    y_original = (y_original_MF + y_original_MI) / 2

    # y_softimpute = softImpute.fit_transform(y_s)
    # y_original = biscaler.inverse_transform(X_imputed)
    test_error = LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))


    # dfcor = pd.DataFrame(y_original)
    # Var_Corr = dfcor.corr()
    # # plot the heatmap and annotation on it
    # sns.heatmap(Var_Corr, vmin=0, vmax=1, xticklabels=Var_Corr.columns, yticklabels=Var_Corr.columns, annot=True)
    # plt.text(0.5, 1.1, "Heat Map" , fontsize = 10)
    # plt.show()

    # test_error = (LA.norm(y[np.isnan(y_miss)] - y_original[np.isnan(y_miss)]) / LA.norm(y[np.isnan(y_miss)])) ** 2
    return test_error

# def MI(y_miss, y):
#     n_imputations = 5
#     XY_completed = []
#     for i in range(n_imputations):
#         imputer = IterativeImputer(max_iter=5, sample_posterior=True, random_state=i)
#         XY_completed.append(imputer.fit_transform(y_miss))
#     completed_mean = np.mean(XY_completed, 0)  # mean of the imputed matrix
#     test_error = LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / np.sqrt(np.sum(np.isnan(y_miss)))
#
#     # test_error_MI = (LA.norm(y[np.isnan(y_miss)] - completed_mean[np.isnan(y_miss)]) / LA.norm(
#     #     y[np.isnan(y_miss)])) ** 2
#     return test_error