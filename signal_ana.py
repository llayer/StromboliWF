import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from minisom import MiniSom    
from matplotlib import cm
from datetime import datetime, date
from scipy.stats import kde



def load_signals(path, wf_len = 259):
    
    df = pd.read_hdf(path)
    
    def check_wf(wf):
        if len(wf) != wf_len:
            return 0
        else:
            return 1  
        
    df['accept_signal'] = df['waveform'].apply(check_wf)
    df = df[df['accept_signal']>0]
    df = df.drop(['accept_signal'], axis=1)
    
    df.time = df.time.astype(str)
    df.time = pd.to_datetime(df['time'])
    
    return df


def exclude_days(df, days):
    
    for day in days:
        mask = ((df['time'] < day) | (df['time'] > day))
        df = df.loc[mask]
    return df


def test_period(df):
    
    period1 = ('2019-06-09', '2019-06-12')
    period2 = ('2019-06-24', '2019-06-27')
    
    mask = ((df['time'] >= period1[0]) & (df['time'] < period1[1])) |((df['time'] >= period2[0]) & (df['time'] < period2[1]))
  
    return df.loc[mask]


def normalize(df, col = 'waveform'):
    
    def scale(vec):
        vec-=np.mean(vec)
        vec/=np.std(vec)
        return vec
    
    df[col] = df[col].apply(scale)
    
    
def plot_freq(df):
    
    plt.figure(figsize=(15,7))
    df['time'].groupby([df["time"].dt.month, df["time"].dt.day]).count().plot(kind="bar")
    plt.show()
    
    
    
def get_features(df, wf=True, lpc=False, amp=False):
    
    def final_input(waveform, lpc_coeff, amplitude):
    
        feat = []
        if wf == True:
            feat += list(waveform)
        if lpc == True:
            feat += list(lpc_coeff)
        if amp == True:
            feat += list(amplitude)
        return feat
        
    features = df.apply(lambda x: final_input(x['waveform'], x['lpc_coeff'], x['amplitude']), axis=1)#.to_numpy()#.values
    features = pd.DataFrame(features.tolist())
    return features.to_numpy()
    
    
def run_som(features, size, niter = 10000, sigma=0.3, learning_rate=.5, pca=True, plot_error = False, random_seed = 1):
    
    som = MiniSom(size, size, features.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed = random_seed) 
    if pca == True:
        som.pca_weights_init(features)
    
    if plot_error == True:
        
        q_error = []
        t_error = []
        iter_x = []
        for i in range(niter):
            percent = 100*(i+1)/niter
            rand_i = np.random.randint(len(features)) # This corresponds to train_random() method.
            som.update(features[rand_i], som.winner(features[rand_i]), i, niter)
            if (i+1) % 100 == 0:
                q_error.append(som.quantization_error(features))
                t_error.append(som.topographic_error(features))
                iter_x.append(i)
        
        plt.plot(iter_x, q_error)
        plt.ylabel('quantization error')
        plt.xlabel('iteration index')
        plt.show()
        
        plt.plot(iter_x, t_error)
        plt.ylabel('topo error')
        plt.xlabel('iteration index')
        plt.show()

    else:
        som.train_random(features, niter) 
    
    return som
    
    
def plot_waveforms(features, som, size):
    
    win_map = som.win_map(features)

    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size, size)
    for position in win_map.keys():
    
        ax = plt.subplot(the_grid[position[0], position[1]])
        for vec in win_map[position]:
            plt.plot(vec, color='gray', alpha=.1)
        plt.plot(np.mean(win_map[position], axis=0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()
    

def plot_dmap(features, som, size):
    
    cmap = cm.get_cmap('bone_r') #gray_r')
    win_map = som.win_map(features)
    
    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size, size)

    dmap = som.distance_map() 
    for i in range(size):
        for j in range(size):

            ax = plt.subplot(the_grid[i, j])
            c = cmap(dmap[i, j])
            ax.set_facecolor((c[0], c[1], c[2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if (i, j) in win_map.keys():
                plt.plot(np.mean(win_map[i, j], axis=0))

    plt.show()
    

def plot_density(features, som, size):
    
    win_map = som.win_map(features)
    
    def density_map():
    
        dm = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                dm[i,j] = len(win_map[i,j])
        return dm/dm.max()
    
    density = density_map()
    
    cmap = cm.get_cmap('viridis') #gray_r')
    
    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size, size)

    for i in range(size):
        for j in range(size):

            ax = plt.subplot(the_grid[i, j])
            c = cmap(density[i, j])
            ax.set_facecolor((c[0], c[1], c[2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if (i, j) in win_map.keys():
                plt.plot(np.mean(win_map[i, j], axis=0))

    plt.show()
    
    
def get_erruption_time():
    
    h = [{'time':'2019-07-04T00:00:00.000000Z'},
     {'time':'2019-08-28T00:00:00.000000Z'} ]
    h = pd.DataFrame(h)
    h.time = h.time.astype(str)
    h.time = pd.to_datetime(h['time'])
    h['time'] = h.time.astype(np.int64)
    err1 = h.iloc[0]['time']
    err2 = h.iloc[1]['time']
    return err1, err2
    
    
def plot_evo(df, som, size):
    
    def to_cluster(waveform):
        return som.winner(waveform)

    def to_index(cluster):
        index = size * cluster[0] + cluster[1]
        return index
    
    df['cluster'] = df['waveform'].apply(to_cluster)
    df['cluster_index'] = df['cluster'].apply(to_index)
    df['time_int'] = df.time.astype(np.int64)

    x, y = df['time_int'], df['cluster_index']

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10,6))
    plt.scatter(x, y, s=1)
    ax.set_xticklabels([datetime.fromtimestamp(ts / 1e9).strftime('%D') for ts in ax.get_xticks()])    
    
         
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    d0 = date(2019, 5, 15)
    d1 = date(2019, 9, 19)
    delta = d1 - d0
    days = delta.days 
    k = kde.gaussian_kde([x,y])
    bins_y = size*size
    bins_x = days
    xi, yi = np.mgrid[x.min():x.max():bins_x*1j, y.min():y.max():bins_y*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        
    # Make the plot
    err1, err2 = get_erruption_time()
    fig, ax = plt.subplots(figsize=(14,10))
    plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
    ax.axvline(x=err1, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(x=err2, color='r', linestyle='dashed', linewidth=2)
    ax.set_xticklabels([datetime.fromtimestamp(ts / 1e9).strftime('%D') for ts in ax.get_xticks()])

    
    plt.show()
    
    