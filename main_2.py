# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # normalize Y at beginning
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
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    import torch.utils.data as Data
    import numpy as np
    import random
    import time
    import datetime
    import seaborn as sns
    # datetime.datetime.now()
    import scipy.stats as stats

    import sklearn.neighbors._base
    import sys

    sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base


    # a = [1.2, 1.5, 1.9]; b = [2.2, 2.5, 3.1]
    # stats.pearsonr(a,b)


    time1 = time.time()

    # softImpute = SoftImpute(max_rank=10)  # simulate data is rank 10!
    # biscaler = BiScaler()
    # biscaler_cmp = BiScaler()


    BATCH_SIZE = 50
    n = 100
    p = 6
    w1 = 1
    # w2 = 1

    # BATCH_SIZE = 250
    # n = 500

    # def repFun(rep):
    rep = 1

    #
    # par_LR = 0.01

    from RMSE2_GCDS_11_04 import *



    def randmis(x, pm):
        if random.uniform(0, 1) < pm:
            return None
        else:
            return x


    def fMCAR(X, pi):
        for j in range(len(X[0])):
            for i in range(len(X)):
                X[i, j] = randmis(X[i, j], pi)
        # return X

    def fMCAR_last_col(X):
        for j in range(len(X[0])):
            for i in range(len(X)):
                if j == len(X[0]) - 1:
                    X[i, j] = None


    def fMAR(X, b0, b1):
        for j in range(len(X[0])):
            for i in range(len(X)):
                pij = 1 / (1 + np.exp(-(b0 + b1 * X[i, j])))
                print(pij)
                X[i, j] = randmis(X[i, j], pij)
        return X



    def sim_data(n = 2000, model_num = 1):
        # n = 2000
        # batch_size = 500
        p = 6

        mu = [0] * p
        sigma_1 = np.diagflat([1.0] * p)
        X = np.random.multivariate_normal(mu, sigma_1, n)

        ### Choose simulation model
        # model_num = 1
        # model_num = 2
        # model_num = 3
        # model_num = 4

        Y_label = 'M' + str(model_num)

        if Y_label == 'M1':
            Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
            Y1_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4])
            Y1_sd = np.ones(n).reshape(-1, 1)
            Y = np.reshape(Y1, (-1, 1))
            Y_u = np.reshape(Y1_u, (-1, 1))

        elif Y_label == 'M2':
            Y2 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4] + (
                    0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2) * X[:, 5]
            Y2_u = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + X[:, 3] - X[:, 4]
            Y2_sd = (0.5 + X[:, 1] ** 2 / 2 + X[:, 4] ** 2 / 2).reshape(-1, 1)
            Y = np.reshape(Y2, (-1, 1))
            Y_u = np.reshape(Y2_u, (-1, 1))

        elif Y_label == 'M3':
            u = np.random.random(n)
            eps = np.random.normal(((u > 0.5) - 0.5) * 2 * 2, 1, n)
            exp_eps = np.exp(eps * 0.5)
            Y3 = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * exp_eps
            Y3_u = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.mean())
            Y3_sd = (5 + X[:, 0] ** 2 / 3 + X[:, 1] ** 2 + X[:, 2] ** 2 + X[:, 3] + X[:, 4]) * (exp_eps.std()).reshape(-1, 1)
            Y = np.reshape(Y3, (-1, 1))
            Y_u = np.reshape(Y3_u, (-1, 1))

        elif Y_label == 'M4':
            # n4 = 1000
            u = np.random.random(n)
            Y4 = np.random.normal(((u > 0.5) - 0.5) * 2 * X[:, 0], 0.25 ** 2, n).reshape(n, -1)
            Y4_u = np.zeros(n)
            # Y4_sd = Y4.std(axis=1).reshape(-1, 1)
            Y = np.reshape(Y4, (-1, 1))
            Y_u = np.reshape(Y4_u, (-1, 1))

            X = X[:, 0].reshape(-1,1)


        ### Gamma
        elif Y_label == 'M5':
            for i in range(p):
                X[:, i] = np.random.gamma(i+1, 1/(i+1), n)

            Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
            Y = np.reshape(Y1, (-1, 1))

        ### Lognormal
        elif Y_label == 'M6':
            for i in range(p):
                X[:, i] = np.random.lognormal(0, (i+1)/5, n)

            Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
            Y = np.reshape(Y1, (-1, 1))

        ### 7
        elif Y_label == 'M7':
            for i in range(p):
                if i == 0:
                    X[:, i] = np.random.lognormal(0, (i+1)/5, n)
                else:
                    X[:, i] = X[:, i-1] ** 2

            Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
            Y = np.reshape(Y1, (-1, 1))

        #
        # ### 8
        # elif Y_label == 'M8':
        #
        #     x1_u = np.random.lognormal(0, 0.625, n)
        #     X[:, 0] = x1_u + np.random.normal(0, 0.5, n)
        #     X[:, 1] = x1_u ** 2 + np.random.normal(0, 0.5, n)
        #
        #     x2_u = np.random.gamma(2, 0.5, n)
        #     X[:, 2] = x2_u + np.random.normal(0, 0.5, n)
        #     X[:, 3] = x2_u ** 2 + np.random.normal(0, 0.5, n)
        #
        #     for i in range(p):
        #         if i == 0:
        #             X[:, i] = np.random.lognormal(0, (i+1)/5, n)
        #         else:
        #             X[:, i] = X[:, i-1] ** 2
        #
        #     Y1 = X[:, 0] ** 2 + np.exp(X[:, 1] + X[:, 2] / 3) + np.sin(X[:, 3] + X[:, 4]) + X[:, 5]
        #     Y = np.reshape(Y1, (-1, 1))

        return pd.DataFrame(np.concatenate((X, Y), axis=1))



    def sim_data_new(n = 2000, model_num = 1):
        # n = 2000
        # batch_size = 500
        p = 6

        mu = [0] * p
        sigma_1 = np.diagflat([1.0] * p)
        X = np.random.multivariate_normal(mu, sigma_1, n)

        ### Choose simulation model
        # model_num = 1
        # model_num = 2
        # model_num = 3
        # model_num = 4

        ### 1
        if model_num == 1:

            x1_u = np.random.lognormal(0, 0.625, n)
            X[:, 0] = x1_u + np.random.normal(0, 0.5, n)
            X[:, 1] = x1_u ** 2 + np.random.normal(0, 0.5, n)

            x2_u = np.random.gamma(2, 0.5, n)
            X[:, 2] = x2_u + np.random.normal(0, 0.5, n)
            X[:, 3] = x2_u ** 2 + np.random.normal(0, 0.5, n)

            X[:, 4] = x1_u + x2_u + np.random.normal(0, 0.5, n)
            X[:, 5] = x1_u * x2_u + np.random.normal(0, 0.5, n)


        ### 1.1
        if model_num == 1.1:

            x1_u = np.random.lognormal(0, 0.625, n)
            X[:, 0] = x1_u
            X[:, 1] = x1_u ** 2

            x2_u = np.random.gamma(2, 0.5, n)
            X[:, 2] = x2_u
            X[:, 3] = x2_u ** 2

            X[:, 4] = x1_u + x2_u
            X[:, 5] = x1_u * x2_u


        ### 1.2
        if model_num == 1.2:

            x1_u = np.random.lognormal(0, 0.625, n)
            X[:, 0] = x1_u
            X[:, 1] = x1_u ** 2

            x2_u = np.random.gamma(2, 0.5, n)
            X[:, 2] = x2_u
            X[:, 3] = x2_u ** 2

            X[:, 4] = x1_u + x2_u
            X[:, 5] = X[:, 4] ** 2


        ### 2
        if model_num == 2:

            x1_u = np.random.lognormal(0, 0.25, n)
            X[:, 0] = x1_u + np.random.normal(0, 0.5, n)
            X[:, 1] = x1_u ** 2 + np.random.normal(0, 0.5, n)

            x2_u = np.random.gamma(1, 1, n)
            X[:, 2] = x2_u + np.random.normal(0, 0.5, n)
            X[:, 3] = x2_u ** 2 + np.random.normal(0, 0.5, n)

            X[:, 4] = x1_u + x2_u + np.random.normal(0, 0.5, n)
            X[:, 5] = x1_u * x2_u + np.random.normal(0, 0.5, n)


            ### 2.1
        if model_num == 2.1:

            x1_u = np.random.lognormal(0, 0.25, n)
            X[:, 0] = x1_u
            X[:, 1] = x1_u ** 2

            x2_u = np.random.gamma(1, 1, n)
            X[:, 2] = x2_u
            X[:, 3] = x2_u ** 2

            X[:, 4] = x1_u + x2_u
            X[:, 5] = x1_u * x2_u



        ### 3.1
        if model_num == 1.1:

            x1_u = np.random.lognormal(0, 0.625, n)
            X[:, 0] = x1_u
            X[:, 1] = x1_u ** 2
            X[:, 4] = x1_u ** 3

            x2_u = np.random.gamma(2, 0.5, n)
            X[:, 2] = x2_u
            X[:, 3] = x2_u ** 2
            X[:, 5] = x2_u ** 3

            # X[:, 4] = x1_u + x2_u
            # X[:, 5] = x1_u * x2_u


        ### 3.2
        if model_num == 1.1:

            x1_u = np.random.lognormal(0, 0.625, n)
            X[:, 0] = x1_u
            X[:, 1] = x1_u ** 2


            x2_u = np.random.gamma(2, 0.5, n)
            X[:, 2] = x2_u
            X[:, 3] = x2_u ** 2

            X[:, 4] = x1_u * x2_u
            X[:, 5] = (x1_u * x2_u) ** 2

        return pd.DataFrame(X)



    # ### Simulation Data
    # df = sim_data(n= 2500*2, model_num = 7)
    #

    model_num = 2.1

    ### Simulation Data New
    df = sim_data_new(n= 2000, model_num = model_num)



    # ###############
    # # Letter Data 0
    # # define the dataset location
    # filename = 'letter-recognition.csv'
    # # filename = 'german.data-numeric'
    # # load the csv file as a data frame
    # # df0 = pd.read_csv(filename, header=None, index_col=0)
    # df0 = pd.read_csv(filename, header=None)
    # df = df0.iloc[:, 1:]
    # # select columns with numerical data types
    #



    ###############
    # Letter Data 0
    # define the dataset location
    # filename = 'testdata6.csv'
    # # filename = 'german.data-numeric'
    # # load the csv file as a data frame
    # # df0 = pd.read_csv(filename, header=None, index_col=0)
    # df0 = pd.read_csv(filename, header=0)
    # df = df0.iloc[:, 1:]
    # select columns with numerical data types



    #
    # ### Credit  1
    # filename = 'default of credit card clients.xls'
    # # filename = 'german.data-numeric'
    # # load the csv file as a data frame
    # df0 = pd.read_excel(filename)
    # df = df0.iloc[1:, 1:-1]
    #
    #
    #
    # # breast  2
    # from sklearn.datasets import load_breast_cancer
    # df0 = load_breast_cancer()
    # df = pd.DataFrame(df0['data'])



    # ###spam   3
    # filename = 'spambase.csv'
    # # filename = 'german.data-numeric'
    # # load the csv file as a data frame
    # df0 = pd.read_csv(filename, header=None)
    # df = df0.iloc[:, :-1]







    # df process
    df = df.astype('float64')

    ## normalize
    data = df.copy()
    df = (data - data.mean()) / data.std()



    # # fMCAR_last_col st
    # train_data_comp = df.iloc[:n, :].to_numpy().copy()
    # train_data_miss = df.iloc[n:2*n, :].to_numpy().copy()
    #
    # fMCAR_last_col(train_data_miss)
    #
    # train_data = np.concatenate((train_data_comp, train_data_miss), axis=0)
    # train_data_true = df.iloc[:2*n, :].to_numpy().copy()
    #
    # y = np.copy(train_data_true)
    # # y_mcar = np.copy(train_data)
    # # fMCAR_last_col end


    y = df.iloc[:n, :].to_numpy()






    # %%####################################

    # 1 MCAR
    # missing_rates = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    missing_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    # missing_rates = [0.1, 0.3, 0.5, 0.7]
    # missing_rates = [0.1, 0.4, 0.7]
    # missing_rates = [0.1, 0.2]
    # fMCAR(y_mcar, k)
    # for k in missing_rates:

    # 2 MCAR2
    a = 5
    b_values = [2, 3, 5, 7, 12, 20, 45][::-1]
    # fMCAR2(y_mcar, a, b)
    # for b in b_values:
    #     print(['b: ', b, a/(a+b)])


    # 3 MAR
    b0 = - 2
    # b1_values = range(7)
    b1_values = [0, 2, 4, 5, 7, 8, 10]

    # fMAR(y_mcar, b0, b1)
    # for b1 in b1_values:
    #     print(['b1: ', b1, 1 / (1 + np.exp(-(b0 + b1 * 0.5)))])

    # 4 MAR2
    u0 = - 2
    # u1_values = range(7)
    u1_values = [0, 2, 4, 5, 7, 8, 10]
    # fMAR2(y_mcar, u0, u1)
    # for u1 in u1_values:
    #     b0j = np.random.normal(loc=u0, scale=1)
    #     b1j = np.random.normal(loc=u1, scale=1)
    #     print(['u1: ', u1, 1 / (1 + np.exp(-(b0j + b1j * 0.5)))])

    MCAR_mean_errs = []
    MCAR_MI_errs = []
    MCAR_MIKNN_errs = []
    MCAR_MF_errs = []
    MCAR_MI_GCDS_errs = []

    MCAR_mean_errs_std = []
    MCAR_MI_errs_std = []
    MCAR_MIKNN_errs_std = []
    MCAR_MF_errs_std = []
    MCAR_MI_GCDS_errs_std = []

    #

    # # def repFun(rep):
    # rep = 1

    for k in missing_rates:
        # for b in b_values:
        # for b1 in b1_values:
        # for u1 in u1_values:

        MCAR_mean_errsl = []
        # MCAR_soft_errsl = []
        # MCAR_con_errsl = []
        MCAR_MI_errsl = []
        MCAR_MIKNN_errsl = []
        MCAR_MF_errsl = []
        MCAR_MI_GCDS_errsl = []
        MCAR_GAIN_errsl = []

        for _ in range(rep):
            # y = np.random.multivariate_normal(mean, sigma_1, n)

            pattern_label = 1
            # pattern_label = 2


            # random missing
            if pattern_label == 1:

                while True:
                    y_mcar = np.copy(y)

                    ## Missing Pattern

                    ## 1 MCAR
                    fMCAR(y_mcar, k)
                    print(['rep, k: ', rep, k])

                    # # 3 MAR
                    # fMAR(y_mcar, b0, b1)
                    # print(['rep, b1: ', rep, b1])


                    if not any(np.isnan(y_mcar).sum(axis=1) == p):
                        break


            # 1 MCAR
            # fMCAR(y_mcar, k)
            # X_cmp = X_ICU_cmp

            ## for M4:
            # y_M4 = y.copy()
            # y[:, 1] = 0


            # GAIN
            time1_mmd = time.time()

            # test_mcar_GAIN = MMD(y_mcar, y, X_cmp, BATCH_SIZE, k, w1, w2)
            # MCAR_GAIN_errsl.append(test_mcar_GAIN)
            # print(['aaa', test_mcar_GAIN])
            # # test_mar_GAIN = GAIN(y_mar, y)
            # # MAR_GAIN_errs.append(test_mar_GAIN)
            #
            # time2_mmd = time.time()
            # print('MMD time: ' + str(datetime.timedelta(seconds=time2_mmd - time1_mmd)))

            # mean imputation
            time1_mean = time.time()

            test_mcar_mean = missmean(y_mcar, y)
            MCAR_mean_errsl.append(test_mcar_mean)

            time2_mean = time.time()
            print('mean time: ' + str(datetime.timedelta(seconds=time2_mean - time1_mean)))

            # # softimpute
            # test_mcar_soft = soft(y_mcar, y)
            # MCAR_soft_errsl.append(test_mcar_soft)
            # # test_mar = soft(y_mar, y)
            # # MAR_soft_errs.append(test_mar)
            #
            # # concat mask
            # test_error_mcar = soft_mask(y_mcar, y)
            # MCAR_con_errsl.append(test_error_mcar)
            # # test_error_mar = soft_mask(y_mar, y)
            # # MAR_con_errs.append(test_error_mar)

            # MI
            time1_MI = time.time()

            test_mcar_MI = MI(y_mcar, y)
            MCAR_MI_errsl.append(test_mcar_MI)
            # test_mar_MI = MI(y_mar, y)11.15hh
            # MAR_MI_errs.append(test_mar_MI)

            time2_MI = time.time()
            print('MI time: ' + str(datetime.timedelta(seconds=time2_MI - time1_MI)))

            # MIKNN
            time1_MIKNN = time.time()

            test_mcar_MIKNN = MIKNN(y_mcar, y)
            MCAR_MIKNN_errsl.append(test_mcar_MIKNN)
            # test_mar_MIKNN = MIKNN(y_mar, y)
            # MAR_MIKNN_errs.append(test_mar_MIKNN)

            time2_MIKNN = time.time()
            print('MIKNN time: ' + str(datetime.timedelta(seconds=time2_MIKNN - time1_MIKNN)))

            # MissForest
            time1_MissFores = time.time()

            test_mcar_MF = MF(y_mcar, y)
            MCAR_MF_errsl.append(test_mcar_MF)
            # test_mar_MF = MF(y_mar, y)
            # MAR_MF_errs.append(test_mar_MF)

            time2_MissFores = time.time()
            print('MissFores time: ' + str(datetime.timedelta(seconds=time2_MissFores - time1_MissFores)))


            # MIGCDS
            time1_MIGCDS = time.time()

            test_mcar_MIGCDS = MI_GCDS_1(y_mcar, y)
            MCAR_MI_GCDS_errsl.append(test_mcar_MIGCDS)
            # test_mar_MF = MF(y_mar, y)
            # MAR_MF_errs.append(test_mar_MF)

            time2_MIGCDS = time.time()
            print('MIGCDS time: ' + str(datetime.timedelta(seconds=time2_MIGCDS - time1_MIGCDS)))





        MCAR_mean_errs.append(sum(MCAR_mean_errsl) / len(MCAR_mean_errsl))
        # MCAR_soft_errs.append(sum(MCAR_soft_errsl) / len(MCAR_soft_errsl))
        # MCAR_con_errs.append(sum(MCAR_con_errsl) / len(MCAR_con_errsl))
        MCAR_MI_errs.append(sum(MCAR_MI_errsl) / len(MCAR_MI_errsl))
        MCAR_MIKNN_errs.append(sum(MCAR_MIKNN_errsl) / len(MCAR_MIKNN_errsl))
        MCAR_MF_errs.append(sum(MCAR_MF_errsl) / len(MCAR_MF_errsl))
        MCAR_MI_GCDS_errs.append(sum(MCAR_MI_GCDS_errsl) / len(MCAR_MI_GCDS_errsl))
        # MCAR_GAIN_errs.append(sum(MCAR_GAIN_errsl) / len(MCAR_GAIN_errsl))

        MCAR_mean_errs_std.append(np.std(MCAR_mean_errsl))
        MCAR_MI_errs_std.append(np.std(MCAR_MI_errsl))
        MCAR_MIKNN_errs_std.append(np.std(MCAR_MIKNN_errsl))
        MCAR_MF_errs_std.append(np.std(MCAR_MF_errsl))
        MCAR_MI_GCDS_errs_std.append(np.std(MCAR_MI_GCDS_errsl))


    # %%



    plt.plot(
        missing_rates, MCAR_mean_errs,
        # missing_rates, MCAR_soft_errs,
        # missing_rates, MCAR_con_errs,
        missing_rates, MCAR_MI_errs,
        missing_rates, MCAR_MIKNN_errs,
        missing_rates, MCAR_MF_errs,
        missing_rates, MCAR_MI_GCDS_errs,
        # missing_rates, MCAR_GAIN_errs,
    )

    version_num = 25
    plt_title = 'RMSE ~ Missing Rates' + ' N: ' + str(n) + ' p: ' + str(p) \
                + ' NUM: ' + str(version_num) + ' ' + str(datetime.datetime.now())
    plt.title(plt_title)
    plt.legend(['Mean Imputation',           'Multiple Imputation',           'Multiple Imputation with KNN',            'MissForest',            'MI_GCDS',
                ], loc='best')
    # plt.ylim((0, 1.2))
    plt.savefig('-'.join([a.strip() for a in plt_title.split(',')]) + '.png')
    plt.show()
    plt.close()






    # datetime.datetime.now()
    # print(datetime.datetime.now())





    time2 = time.time()
    time2 - time1
    print('Time used: ' + str(datetime.timedelta(seconds=time2 - time1)))

    # print('y_var: ' + str(np.mean(y_var)))


    # LA.norm(2*np.zeros([5,5]) - 2*np.ones([5,5]))
    #
    # / np.sqrt(np.sum(np.isnan(y_miss)))

    #
    #
    # ## 2021.10.28 with GCDS
    # MCAR_GCDS_errs = []
    #
    # plt.plot(
    #     missing_rates, MCAR_mean_errs,
    #     # missing_rates, MCAR_soft_errs,
    #     # missing_rates, MCAR_con_errs,
    #     missing_rates, MCAR_MI_errs,
    #     missing_rates, MCAR_MIKNN_errs,
    #     missing_rates, MCAR_MF_errs,
    #     # missing_rates, MCAR_GCDS_errs,
    # )
    # plt_title = 'RMSE ~ Missing Rates, Letter' + ' N: ' + str(n) + ' p: ' + str(p) \
    #             + ' BATCH: ' + str(BATCH_SIZE) + \
    #             ' w2: ' + str(w2) + \
    #             ' lr: ' + str(par_LR) + \
    #             ' rep: ' + str(rep) + ' ' + str(datetime.datetime.now())
    # plt.title(plt_title)
    # plt.legend(['Mean Imputation',
    #             'Multiple Imputation',
    #             'Multiple Imputation with KNN',
    #             'MissForest',
    #             'GCDS',
    #             ], loc='best')
    # plt.savefig('-'.join([a.strip() for a in plt_title.split(',')]) + '.png')
    # plt.show()
    #
    # plt.close()

    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
