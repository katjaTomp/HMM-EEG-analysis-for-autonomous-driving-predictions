import pandas as pd
import numpy as np
import os

from sklearn import cluster
from hmmlearn.hmm import GaussianHMM,GMMHMM

#from xlrd import open_workbook

# Omit warnings

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


data_location = "folder"

# Pickled file names
def loadData(electrode="FCz"):
    """
    This function prepares the data for training.
    If no  electrode is defined the FCz electrode's values are only considered

    :param electrode: string which can be any valid electrode name
    :return: list of vectors containing from an electrode
    """

    frontal = []

    filenames = [f for f in os.listdir(data_location) if f.endswith('.pkl')]

    frontal_final = [np.array(pd.read_pickle(filename)[electrode]).reshape(-1, 1) for filename in filenames[:-2]]

    participant_19 = np.concatenate(
            [np.array(pd.read_pickle(filename)[electrode]).reshape(-1, 1) for filename in filenames[-2:]])

    frontal_final.append(participant_19)

    file_lengths = [len(x) for x in frontal_final]

    return [frontal_final, file_lengths]



def getTestingTrainingSets(X, lengths):
    """
    The sets yielded are used for cross validation
    :param frontal_final:
    :param lengths:
    :return:
    """

    j = 0
    sum = 0

    for length in lengths:
        fromm = sum
        to = sum + length

        if sum == 0:

            yield [X[fromm:to],X[to:],lengths[j+1:]]

        else:
            lengths_train = []
            test_set = X[fromm:to]
            sub_train_1 = X[0:fromm]
            sub_train_2 = X[to:]

            for i in range(0,17):

                if j!=i:
                    lengths_train.append( lengths[i] )

            train_set = np.concatenate([sub_train_1, sub_train_2])

            yield [test_set,  train_set,lengths_train]

        sum = sum + length
        j = j +1




def trainModel(X,lengths,states):
    model = GaussianHMM(n_components=states, covariance_type="full", n_iter=1000).fit(X, lengths)

    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    score = model.score(X,lengths)

    print(score)
    return [model,score]


def trainModelGMM(X, lengths, states, num_gaus):

    model = GMMHMM(n_components=states, n_mix=num_gaus,n_iter=1000,verbose=True).fit(X,lengths)

    print('Mixture Models + HMM')
    print(model.predict(X))
    print(model.monitor_.converged)
    print(model.monitor_)
    print(model.score(X, lengths))


def saveModel(model,num_state):

     from sklearn.externals import joblib
     joblib.dump(model, "model_"+str(num_state)+"_200.pkl")

datas = loadData()
print(len(datas[0]), datas[1])

crossValidation = getTestingTrainingSets(datas[0],datas[1])
for states in range(1, 12):


    print("states number:", states)
    sum_score = 0
    score_model_best = 100000000

    for test, train,train_length in getTestingTrainingSets(datas[0], datas[1]):

        result = trainModel(train,train_length, states)
        model = result[0]
        score_model = result[1]
        if score_model < score_model_best:
            score_model_best = score_model
            saveModel(model, states)

        score = model.score(test)
        print("Score:", model.score(test))
        sum_score +=score

    print("Mean score:",sum_score/18)


