import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib
matplotlib.use('TkAgg')
matplotlib.interactive(False)
from matplotlib import cm, pyplot as plt
import pickle


def getEvents(ppn):
    epochs_time_events = pickle.load(open("ppn"+str(ppn)+"_events"))
    return epochs_time_events


def loadData(t, condition,electrode="FCz"):

    frontal = []
    if ((t == False) & (condition == 'both')):

        participants_data_1 = 'noEOG_diff_epoching/1.pkl'
        participants_data_2 = 'noEOG_diff_epoching/2.pkl'
        participants_data_3 = 'noEOG_diff_epoching/3.pkl'
        participants_data_4 = 'noEOG_diff_epoching/4.pkl'
        participants_data_5 = 'noEOG_diff_epoching/5.pkl'
        participants_data_6 = 'noEOG_diff_epoching/6.pkl'
        participants_data_7 = 'noEOG_diff_epoching/7.pkl'
        participants_data_9 = 'noEOG_diff_epoching/9.pkl'
        participants_data_10 = 'noEOG_diff_epoching/10.pkl'
        participants_data_11 = 'noEOG_diff_epoching/11.pkl'
        participants_data_12 = 'noEOG_diff_epoching/12.pkl'
        participants_data_13 = 'noEOG_diff_epoching/13.pkl'
        participants_data_14 = 'noEOG_diff_epoching/14.pkl'
        participants_data_15 = 'noEOG_diff_epoching/15.pkl'
        participants_data_16 = 'noEOG_diff_epoching/16.pkl'
        participants_data_17 = 'noEOG_diff_epoching/17.pkl'
        participants_data_18 = 'noEOG_diff_epoching/18.pkl'
        participants_data_19_1 = 'noEOG_diff_epoching/19_1.pkl'
        participants_data_19_2 = 'noEOG_diff_epoching/19_2.pkl'

        frontal_1 = np.array(pd.read_pickle(participants_data_1)[electrode]).reshape(-1, 1)
        frontal_2 = np.array(pd.read_pickle(participants_data_2)[electrode]).reshape(-1, 1)
        frontal_3 = np.array(pd.read_pickle(participants_data_3)[electrode]).reshape(-1, 1)
        frontal_4 = np.array(pd.read_pickle(participants_data_4)[electrode]).reshape(-1, 1)
        frontal_5 = np.array(pd.read_pickle(participants_data_5)[electrode]).reshape(-1, 1)
        frontal_6 = np.array(pd.read_pickle(participants_data_6)[electrode]).reshape(-1, 1)
        frontal_7 = np.array(pd.read_pickle(participants_data_7)[electrode]).reshape(-1, 1)
        frontal_9 = np.array(pd.read_pickle(participants_data_9)[electrode]).reshape(-1, 1)
        frontal_10 = np.array(pd.read_pickle(participants_data_10)[electrode]).reshape(-1, 1)
        frontal_11 = np.array(pd.read_pickle(participants_data_11)[electrode]).reshape(-1, 1)
        frontal_12 = np.array(pd.read_pickle(participants_data_12)[electrode]).reshape(-1, 1)
        frontal_13 = np.array(pd.read_pickle(participants_data_13)[electrode]).reshape(-1, 1)
        frontal_14 = np.array(pd.read_pickle(participants_data_14)[electrode]).reshape(-1, 1)
        frontal_15 = np.array(pd.read_pickle(participants_data_15)[electrode]).reshape(-1, 1)
        frontal_16 = np.array(pd.read_pickle(participants_data_16)[electrode]).reshape(-1, 1)
        frontal_17 = np.array(pd.read_pickle(participants_data_17)[electrode]).reshape(-1, 1)
        frontal_18 = np.array(pd.read_pickle(participants_data_18)[electrode]).reshape(-1, 1)
        frontal_19_1 = np.array(pd.read_pickle(participants_data_19_1)[electrode]).reshape(-1, 1)
        frontal_19_2 = np.array(pd.read_pickle(participants_data_19_2)[electrode]).reshape(-1, 1)
        frontal_19 = np.concatenate([frontal_19_1,frontal_19_2])

        frontal_final = [frontal_1,
                         frontal_2,
                         frontal_3,
                         frontal_4,
                         frontal_5,
                         frontal_6,
                         frontal_7,
                         frontal_9,
                         frontal_10,
                         frontal_11,
                         frontal_12,
                         frontal_13,
                         frontal_14,
                         frontal_15,
                         frontal_16,
                         frontal_17,
                         frontal_18,
                         frontal_19]

        lengths = [len(x) for x in frontal_final]


        return [frontal_final, lengths]


def get_ModelInfo(model,X):

    A = model.transmat_
    means = model.means_
    #pi = model.startprob_
    #B = model.predict_proba(X)
    covars = model.covars_

    return means, covars, A


def get_300after_stimulus(events, seq_states):
    prev_epoch = -1
    df = pd.DataFrame(columns=['event', 'time', 'state'])
    entry = 0
    offset = 20
    for[epoch, time, event] in events:
        if epoch !=prev_epoch:
            if time == 0:
                t=0

                if (epoch == 0):
                    #print epoch,event, seq_states[60:80]

                    for state in seq_states[20:80]:
                        data = [event, t, state]
                        df.loc[entry] = data

                        t += 1
                        entry += 1

                else:
                    #print epoch,event,offset, seq_states[offset+60:offset+80]
                    for state in seq_states[offset+60:offset+80]: # 60 account to 300 ms
                        data = [event, t, state]
                        df.loc[entry] = data

                        t += 1
                        entry += 1


                prev_epoch = epoch

        offset += 1

    return df



def get_100before_Epoch(events, seq_states):

    it = 0
    prev_epoch = -1
    #all_events = ["65304","65306","65308","65281","65284","65286","65288","65298","65296","65294"]
    df = pd.DataFrame(columns=['event', 'time', 'state'])
    entry = 0
    offset = 0
    print events.shape

    for [epoch, time, event] in events:

        if epoch != prev_epoch:
            if time == 0:
                t = 0

                if(epoch == 0):

                    for state in seq_states[0:20]:
                        data = [event, t, state]
                        df.loc[entry] = data
                        # print df
                        t += 1
                        entry += 1

                else:
                    for state in seq_states[offset-20:offset]:
                        data = [event, t, state]
                        df.loc[entry] = data

                        t += 1
                        entry += 1

                prev_epoch = epoch

        offset += 1
    print 'klaar'
    return df

def get_early_results(events, seq_states, X,ppn):

    epochs_total = [640, 1516, 1755, 770, 644, 1066, 1209, 797, 1102, 1091, 259, 803]
    it = 0
    prev_epoch = -1
    # all_events = ["65304","65306","65308","65281","65284","65286","65288","65298","65296","65294"]
    df = pd.DataFrame(columns=['event', 'time', 'state'])
    entry = 0
    offset = 0

    for [epoch, time, event] in events:
        #print(epochs_total[ppn]/2)

        if epoch != prev_epoch and epoch > epochs_total[ppn]/2:

            if time == 0:
                t = 0
                # print event

                if (epoch == 0):

                    for state in seq_states[0:20]:
                        data = [event, t, state]
                        df.loc[entry] = data
                        # print df
                        t += 1
                        entry += 1


                else:

                    # print offset
                    for state in seq_states[offset - 20:offset]:
                        data = [event, t, state]
                        df.loc[entry] = data

                        t += 1
                        entry += 1

                    plt.figure(1)

                prev_epoch = epoch

        offset += 1
        # print df
    return df


def get_participants(divide=False, condition='all'):
    """
    If divide is True then the participants are divided into groups of high susceptibility and low susceptibility
    :param divide:
    :return: participants indices
    """
    if condition == 'all':
        if divide:
            participants_high = [1,2,9,11,15,16,17,19]

            participants_low = [3,4,5,6,7,10,12,13,14,18]

            participants = {1: participants_high, 2:participants_low}
        else:

            participant = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19]

            participants = {3:participant}

        return participants

    elif condition =='driving':

        participants_high = [1,2,6,9,11,16,17]
        participants_low = [3,4,5,7,10,12,13,14,15,18,19]


        return {1: participants_high, 2: participants_low}

    elif condition == 'autonomous':
        participants_high = [7,9,11,14,15,17]
        participants_low = [1,2,3,4,5,6,10,12,13,16,18,19]


        return {1: participants_high, 2: participants_low}

    else:
        participants_high = [1,2,9,11,13,15,16,17,19 ]
        participants_low = [3,4,5,6,7,10,12,14,18]



        return {1: participants_high, 2: participants_low}




data =loadData(False,'both',electrode="Fp1")

X, lengths = data[0], data[1]
#FCz ->"model_3_200.pkl"
model = 'model_4fp1_200.pkl'


model = joblib.load(open(model, 'r'))
means, covars, A = get_ModelInfo(model, X)
print means , covars
for a in A:
    print a[0],a[1],a[2]#,a[3],a[4]


df_main = pd.DataFrame(columns=['event', 'time', 'state'])

data = loadData(False, "both")
condition = 'driving'
participants = get_participants(False,condition)

for key,ppn_list in participants.iteritems():
    for i in ppn_list:
        print i
        if i < 9:
            X, lengths = data[0][i - 1], data[1][i - 1]
            events = getEvents(i)

        else:
            X, lengths = data[0][i-2], data[1][i-2]

            events = getEvents(i)

        if len(events) != len(X):
            print "Wrong lengths!"
            break


        #decode the sequences
        sequence_of_states = model.predict(X)



        #df = get_100before_Epoch(events, sequence_of_states)
        df = get_300after_stimulus(events, sequence_of_states)
        #df = getEarlyResults(events, sequence_of_states,X, i)

        df_main = df_main.append(df)



# to change the names accordingly
    if key == 1:
        pickle.dump(df_main, open('dataFrame_High_4_FP1_High_after'+condition, 'w'))
    elif key == 2:
        pickle.dump(df_main, open('dataFrame_Low_4_FP1_Low_after'+condition, 'w'))
    else:
        pickle.dump(df_main, open('dataFrame_ALL_4_FP1_after', 'w'))








