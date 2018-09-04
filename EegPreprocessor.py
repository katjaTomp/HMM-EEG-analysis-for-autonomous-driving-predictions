"""
This script functionality accounts to reading EEG files,removing different types of artifacts by
applying filters and rejecting explicitely annotated EOG events. Next, the data is baseline corrected and
downsampled from 2048 Hz to 200 Hz
"""

import mne
import matplotlib
import numpy as np

import pandas

#### Set up matplotlib ####
matplotlib.use('TkAgg')
matplotlib.interactive(False)


### fix the 50 ms delay by subtracting the 50ms to the events first column's  values
## mne converts seconds to secs * sample freq therefore, I will add 0.05 sec * 2048 Hz = 102,4 ~ 102 to each timestamp in the events

def subtract50ms(events):

    for i in range(len(events)):
        events[i][:1:1] -= 102

    return events


#### Load data ####

def loadFile(file):
    raw = mne.io.read_raw_edf(file, eog=['EXG3', 'EXG4'], stim_channel=-1, preload=True)  # documentation here:https://martinos.org/mne/stable/generated/mne.io.read_raw_edf.html#mne.io.read_raw_edf
    print(raw.plot(block=True,show_options=True))

    findEvents(raw)
    #print(raw.plot(block=True))

    return raw


### Bad Channels are included here ####

def addBadChannels(raw):
    raw.info['bads'] = ['EXG7', 'EXG8']
    return raw


### Events ####

def findEvents(raw):
    events = mne.find_events(raw, shortest_event=1 , output="onset",consecutive='increasing',  stim_channel="STI 014")#,
    events = subtract50ms(events)
    #print(events)

    return events


### Find picks ###

def getPicks(raw):
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True)
    return picks


###filters applied

def highPassfilter(raw, picks, l_freq):
    raw = raw.filter(l_freq=l_freq, h_freq=None, picks=picks)
    return raw


def lowPassfilter(raw, picks, h_freq):
    raw = raw.filter(h_freq=h_freq, l_freq=None, picks=picks)
    return raw


## average reference ###
# Data were referenced to the average of left and right mastoid signal.

def averRef(raw):
    raw = raw.set_eeg_reference(ref_channels='average', projection=False)

    return raw


def eogRef(raw):
    raw = raw.set_eeg_reference(ref_channels=['EXG1', 'EXG2'], projection=False)
    return raw


def genEpochs(raw, events, picks):

    event_id, tmin, tmax =  None, -0.10, 2.00  # was=0 0.5 make it 2.00
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks, baseline=None,
                        reject_by_annotation=True,
                        preload=True)
    return epochs


def applyNotchFilter(raw):
    raw.notch_filter(freqs=50)
    return raw


def rejectEOG(raw):
    eog_events = mne.preprocessing.find_eog_events(raw)
    n_blinks = len(eog_events)
    # Center to cover the whole blink with full duration of 0.5s:
    onset = eog_events[:, 0] / raw.info['sfreq'] - 0.25
    duration = np.repeat(0.5, n_blinks)
    raw.annotations = mne.Annotations(onset, duration, ['bad blink'] * n_blinks,
                                      orig_time=raw.info['meas_date'])
    print(raw.annotations)  # to get information about what annotations we have
    raw.plot(events=eog_events, block=True)
    return raw



def baseline(epochs, time):
    epochs.apply_baseline(baseline=(0, time), verbose=True)

    return epochs


## reducing the sampling frequency

def downsampleEpochs(freq, epochs):
    epochs_resampled = epochs.copy().resample(freq, npad='auto')

    return epochs_resampled


# Data pre-processing
# uncomment the code below and adjust if necessary the bdf file names.
'''

filenames = ['PPN1_10mei.bdf',
             'PPN2_10mei.bdf',
             'PPN3_12mei.bdf',
             'PPN4_18mei.bdf',
             'PPN5_18mei.bdf',
             'PPN6_19mei.bdf',
             'PPN7_19mei.bdf',
             'PPN9-20mei.bdf',
             'PPN10-20mei.bdf',
             'PPN11_20mei.bdf',
             'ppn12_23mei.bdf',
             'PPN13_24MEI.bdf',
             'PPN14-25mei.bdf',
             'PPN15-25MEI.bdf',
             'ppn16-26mei.bdf',
             'ppn17-26mei.bdf',
             'PPN18-31mei.bdf',
             'PPN19-1juni.bdf',
             'PPN19-1juni(2).bdf'

]
'''
filenames = ['PPN19-1juni(2).bdf']
i = 1
for filename in filenames:
    raw = loadFile(filename)

    raw = addBadChannels(raw)  # channels with no signal (to check for all participants)

    raw = applyNotchFilter(raw)  # 50 Hz notch filter
    picks = getPicks(raw)
    raw = highPassfilter(raw, picks, 0.16)  # High pass filter 0.16 Hz
    raw = lowPassfilter(raw, picks, 30)  # Low pass filter 30 Hz

    raw = rejectEOG(raw)
    raw = eogRef(raw)  # reference data to two mastoids

    #print(raw.plot(block=True))

    #### Epoching ####
    events = findEvents(raw)
    print("events:", len(events))

    epochs = genEpochs(raw, events, picks)
    epochs = baseline(epochs,
                  0.1)  # The data were baseline corrected over the 100 ms interval preceding the stimulus presentation.

    epochs = downsampleEpochs(200, epochs)
    print("epochs after downsampling:", epochs)

    epochs.plot(picks,block=True)
    data = epochs.get_data()

    #mne.viz.plot_events(events=events)


    # putting epochs into data frame
    index = ['epoch','time']  # 1, dict(grad=1e13)
    df = epochs.to_data_frame(picks=None, scalings=None, index=index)
    df.to_pickle(filename+'-downsampled_noEOG_200.pkl')
    #i += 1




