"""
This script generates the event list for each of participants
"""
import pandas as pd
import pickle
import sys
import warnings
import os
import glob
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")


# These are the files obtained after running EegPreprocessor.py script
# Adjust the names and location according to your file structure

files = ['noEOG_diff_epoching/1.pkl',
         'noEOG_diff_epoching/2.pkl',
         'noEOG_diff_epoching/3.pkl',
         'noEOG_diff_epoching/4.pkl',
         'noEOG_diff_epoching/5.pkl',
         'noEOG_diff_epoching/6.pkl',
         'noEOG_diff_epoching/7.pkl',
         'noEOG_diff_epoching/9.pkl',
         'noEOG_diff_epoching/10.pkl',
         'noEOG_diff_epoching/11.pkl',
         'noEOG_diff_epoching/12.pkl',
         'noEOG_diff_epoching/13.pkl',
         'noEOG_diff_epoching/14.pkl',
         'noEOG_diff_epoching/15.pkl',
         'noEOG_diff_epoching/16.pkl',
         'noEOG_diff_epoching/17.pkl',
         'noEOG_diff_epoching/18.pkl']


# Only for participant 19th

participants_data_1 = 'noEOG_diff_epoching/19_1.pkl'
participants_data_2 =  'noEOG_diff_epoching/19_2.pkl'

ppn1 = pd.read_pickle(participants_data_1)
ppn1_1 = pd.read_pickle(participants_data_2)
ppn = pd.concat([ppn1,ppn1_1])

result = ppn.reset_index()
result = result[['epoch', 'time', 'condition']]

result = result.as_matrix()

ppn1 = ppn[['condition']]
pickle.dump(result, open('ppn19_events', 'w'))

# For the rest of participants
for file,i in zip(files, range(1,19)):
    ppn = pd.read_pickle(file)
    print ppn
    result = ppn.reset_index()
    result = result[['epoch','time','condition']]
    result = result.as_matrix()

    print result
    ppn1 = ppn[['condition']]
    print ppn1


    pickle.dump(result, open('ppn'+str(i)+'_events', 'w'))


ppn1_array = np.array(ppn1)
















