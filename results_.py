from __future__ import division
import numpy as np
import pickle
import csv
import pandas as pd


#data frames used in analysis
#General 3state analysis
## dataFrame_ALL_3_
## dataFrame High_3
## dataFrame_High_3__driving
## dataFrame_High_3__autonomous
## dataFrame_High_3__stationary
## dataFrame_High_3__afterdriving
## dataFrame_High_3__afterautonomous
## dataFrame_High_3__afterstationary
## dataFrame_Low_3
## dataFrame_Low_3__driving
## dataFrame_Low_3__autonomous
## dataFrame_Low_3__stationary
## dataFrame_Low_3__afterdriving
## dataFrame_Low_3__afterautonomous
## dataFrame_Low_3__afterstationary

#General 5 state analysis

## dataFrame_ALL_5
## dataFrame High_5
## dataFrame_High_5__driving
## dataFrame_High_5__autonomous
## dataFrame_High_5__stationary
## dataFrame_High_5__afterdriving
## dataFrame_High_5__afterautonomous
## dataFrame_High_5__afterstationary
## dataFrame_Low_5
## dataFrame_Low_5__driving
## dataFrame_Low_5__autonomous
## dataFrame_Low_5__stationary
## dataFrame_Low_5__afterdriving
## dataFrame_Low_5__afterautonomous
## dataFrame_Low_5__afterstationary

all_events = ["65304", "65306","65308","65281","65284","65286","65288","65298","65296","65294"]
driving = ["65304", "65306","65308"]
autonomous = ["65298","65296","65294"]
stationary = ["65284","65286","65288"]


probabilities = np.zeros(shape=(10,5))

def get_sub_dataframe(event, df):

    if isinstance( event,str):
        return df.loc[(df['event'] == event)]

    else:

        return df.loc[(df['event'] == event[0]) | (df['event'] == event[1]) | (df['event'] == event[2])]


def get_state_occurences(df_cond_event,state):

    sub_df = df_cond_event.loc[df_cond_event['state'] == state]

    return len(sub_df)

def get_event(condition):

    if condition == 'driving':
        return ["65304", "65306", "65308"]

    elif condition == 'autonomous':

        return ["65294", "65296", "65298"]
    else:
        return ["65284", "65286", "65288"]

def get_data_frame(condition='driving', states="3",ba=True,group='High'):

    if ba:
        ba = ''
    else:
        ba = 'after'
    if states == "3":

        return 'dataFrame_'+group+'_'+ states+ '__' + ba+condition
    else:
        return 'dataFrame_' + group + '_' + states + '_' + ba + condition




states = 3
per_condition =True
ba=True # if False then after stimulus else before stimulus
group='High'
data_frame = pd.DataFrame(columns=['condition','event','percentage'])
all = True



if per_condition:
    #Per condition
    driving_condition = ['driving', 'autonomous', 'stationary']
    for condition in driving_condition:
        filename = get_data_frame(condition=condition, states=str(states), ba=ba,group=group)
        #print filename
        df = pickle.load(open(filename, 'r'))  ## choose the data frame from the list above

        events = get_event(condition)
        df_cond_event = get_sub_dataframe(events, df)
        total_rows = len(df_cond_event)
        instances = []

        for i in range(0, states):

            instances.append(get_state_occurences(df_cond_event,i))

        percentages = [x / total_rows for x in instances]
        print percentages


else:
    # Per event
    j = 0
    driving_condition = ['driving', 'autonomous', 'stationary']

    for event in all_events:
        if event in driving:
            condition = 'driving'
        elif event in autonomous:
            condition = 'autonomous'
        else:
            condition = 'stationary'
        filename = get_data_frame(condition=condition, states=str(states), ba=ba, group=group)
        print filename
        df = pickle.load(open(filename, 'r'))

        events = event
        df_cond_event = get_sub_dataframe(events, df)

        total_rows = len(df_cond_event)

        instances = []

        for i in range(0, states):

            instances.append(get_state_occurences(df_cond_event, i))

        percentages = [x / total_rows for x in instances]
        print event, condition, ba
        print percentages
        # probabilities[j] = percentages
        j+=1


