import obspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from obspy import read
import librosa
import glob

in_path = "/home/llayer/Data/ascii_30/"
out_path = "/home/llayer/Data/Input/"
sps = 50
ascii_files = glob.glob(in_path + "*/*/*/*.ascii")


def param_amplitude(data):
    
    start = 0
    end = sps
    
    amplitudes = []
    n_sec = 0
    
    while end <= len(data):
        
        batch = data[start:end]
        amp = max(batch) - min(batch)
        amplitudes.append(amp)
        
        start = end
        end += sps
        n_sec +=1
        
    amps_normed = []
    for amp in amplitudes:
        amps_normed.append( (amp * n_sec) / sum(amplitudes) )
        
    return amps_normed


def lpc(data, window = 256, overlap = 128, order=6):
    
    startindex = 0
    endindex = window

    features = []
    
    while endindex <= len(data):
                
        y = data[startindex : endindex]
        lpc = librosa.lpc(y, order)
        features.append( lpc )
        
        startindex = endindex - overlap
        endindex = startindex + window
        
    return features
    

def filter_sample(tr, factor=6):

    # Filtering with a lowpass on a copy of the original Trace
    tr_filt = tr.copy()
    tr_filt.filter( 'bandpass', freqmin = 0.05, freqmax = 0.5, corners=2, zerophase=True)
    
    tr_new = tr_filt.copy()
    tr_new.decimate(factor=factor, strict_length=False, no_filter=True)
    return tr_new.data
    
    
def parametrization(file, verbose=0):
    
    st = read(file)
    tr = st[0]
    
    if len(tr.data) < 1:
        return None
    
    tr = tr.detrend() # TODO: Check if necessary
    
    if verbose > 0:
        print( len(tr.data) )
    
    # Filter and sample the data
    waveform = filter_sample(tr)
    if verbose > 0:
            print( len(waveform) )

    # LPC analysis
    lpc_coeff = lpc(tr.data)
    lpc_coeff = np.concatenate(lpc_coeff).ravel()
    if verbose > 0:
            print( len(lpc_coeff) )

    # Amplitude parametrization
    amplitude = param_amplitude(tr.data)
    if verbose > 0:
            print( len(amplitude) )
    
    return {'time': tr.stats.starttime, 'waveform': waveform, 'lpc_coeff':lpc_coeff, 'amplitude':amplitude}


def test(iFile = 150):
    
    print('Start')
    features = parametrization(ascii_files[iFile], verbose=1)
    df = pd.DataFrame([features])
    print(df.head())
    df.to_hdf('test.h5', 'frame')
    

def param_all():
    
    events = []
    corr_files = 0
    for counter, file in enumerate(ascii_files):
        
        if counter%100 == 0:
            print( counter )
        try:
            features = parametrization(file)
            if features is not None:
                events.append(features)
            else:
                corr_files += 1
        except:
            print(file)
            corr_files += 1

    print( 'Number of bad events' , corr_files)
    print( 'Number of good events' , len(events))

    df = pd.DataFrame(events)
    print( df.head() )
    df.to_hdf(out_path + 'features_30.h5', 'frame') 

#test()

param_all()
    
    
    
    
    
    
    
    
