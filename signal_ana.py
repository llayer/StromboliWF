import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from minisom import MiniSom    
from matplotlib import cm
from datetime import datetime, date
from scipy.stats import kde
import skopt
from skopt.utils import use_named_args
from skopt import gp_minimize
import math
from sklearn.manifold import TSNE  
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from random import randint


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
        mask = ((df['time'] < day[0]) | (df['time'] > day[1]))
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
    
    
    
def get_features(df, col, scale_global = True):
        
    features = np.array(df[col].tolist())
    
    if scale_global == True:
        
        shape = features.shape
        features_flat = features.flatten()
        mean = np.mean(features_flat)
        std = np.std(features_flat)
        features_flat = features_flat - mean
        features_flat /= std
        features = features_flat.reshape(shape[0], shape[1])
    
    return features
    
    
def run_som(features, size_x, xize_y, niter = 10000, sigma=0.3, learning_rate=.5, pca=True, plot_error = False, random_seed = 1):
    
    som = MiniSom(size_x, xize_y, features.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed = random_seed) 
    if pca == True:
        som.pca_weights_init(features)
    else:
        som.random_weights_init(features)
    
    if plot_error == True:
        
        q_error = []
        t_error = []
        iter_x = []
        for i in range(niter):
            percent = 100*(i+1)/niter
            rand_i = np.random.randint(len(features)) # This corresponds to train_random() method.
            som.update(features[rand_i], som.winner(features[rand_i]), i, niter)
            if (i+1) % 1000 == 0:
                q_error.append(som.quantization_error(features))
                print( q_error[-1] )
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
    
    
def plot_waveforms(features, som, size_x, size_y):
    
    win_map = som.win_map(features)

    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size_x, size_y)
    for position in win_map.keys():
    
        ax = plt.subplot(the_grid[position[0], position[1]])
        for vec in win_map[position]:
            plt.plot(vec, color='gray', alpha=.1)
        plt.plot(np.mean(win_map[position], axis=0))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    
    plt.show()
    

def plot_dmap(features, som, size_x, size_y):
    
    cmap = cm.get_cmap('bone_r') #gray_r')
    win_map = som.win_map(features)
    
    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size_x, size_y)

    dmap = som.distance_map() 
    for i in range(size_x):
        for j in range(size_y):

            ax = plt.subplot(the_grid[i, j])
            c = cmap(dmap[i, j])
            ax.set_facecolor((c[0], c[1], c[2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            if (i, j) in win_map.keys():
                plt.plot(np.mean(win_map[i, j], axis=0))

    plt.show()
    

def plot_density(features, som, size_x, size_y):
    
    win_map = som.win_map(features)
    
    def density_map():
    
        dm = np.zeros((size_x, size_y))
        for i in range(size_x):
            for j in range(size_y):
                dm[i,j] = len(win_map[i,j])
        return dm/dm.max()
    
    density = density_map()
    
    cmap = cm.get_cmap('viridis') #gray_r')
    
    plt.figure(figsize=(14, 14))
    the_grid = GridSpec(size_x, size_y)

    for i in range(size_x):
        for j in range(size_y):

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
    

def feat_to_cluster(df, som, size_x, size_y, wf = True, lpc = True, amp = True):

    def to_cluster(waveform, lpc_coeff, amplitude):
        
        feat = []
        if wf == True:
            feat += list(waveform)
        if lpc == True:
            feat += list(lpc_coeff)
        if amp == True:
            feat += list(amplitude)
        return som.winner(np.array(feat))

    def to_index(cluster):
        index = size_y * cluster[0] + cluster[1]
        return index
    
    df['cluster'] = df.apply(lambda x: to_cluster(x['waveform'], x['lpc_coeff'], x['amplitude']), axis=1)
    df['cluster_index'] = df['cluster'].apply(to_index)
    df['time_int'] = df.time.astype(np.int64)
    
    
def plot_evo(df, col, n_clusters):

    x, y = df['time_int'], df[col]

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
    bins_y = n_clusters
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


def plot_hist2d(df, col, n_clusters, timebins=200):

    df['time_int'] = df.time.astype(np.int64)
    x, y = df['time_int'], df[col]

    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(timebins, n_clusters))
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]+1]
    
    # 2d histo
    fig = plt.figure(1, figsize=(15, 10))
    ax = fig.add_subplot(111)
    plt.imshow(heatmap.T, extent=extent, aspect='auto', interpolation = 'nearest')

    err1, err2 = get_erruption_time()
    ax.axvline(x=err1, color='r', linestyle='dashed', linewidth=2)
    ax.axvline(x=err2, color='r', linestyle='dashed', linewidth=2)
    ax.set_xticklabels([datetime.fromtimestamp(ts / 1e9).strftime('%D') for ts in ax.get_xticks()])
    
    plt.colorbar()
    plt.show()

    # Fractions in bin
    frac_per_bin = []
    for b in heatmap:
        fraction = []
        for cluster in b:
            if sum(b) > 0:
                fraction.append(float(cluster) / float(sum(b)))
            else:
                fraction.append(0.)
        frac_per_bin.append(fraction)
    
    fig = plt.figure(1, figsize=(12, 8))
    ax = fig.add_subplot(111)
    frac_per_bin = np.array(frac_per_bin)
    for i in range(frac_per_bin.shape[1]):
        vec = frac_per_bin[:,i]
        plt.plot(vec, linewidth=2)
    plt.show()
    
def optimize( dimensions, initial_param, features, size_x, xize_y, num_calls=12, pca = False, random_seed = 1): 

    prior_values = []
    prior_names = []
    for var in dimensions:
        name = var.name
        print( name )
        prior_names.append(name)
        prior_values.append(initial_param[name])

    global num_skopt_call
    num_skopt_call = 0
    errors = []

    @use_named_args(dimensions)
    def fitness(**p): 

        global num_skopt_call

        print('\n \t ::: {} SKOPT CALL ::: \n'.format(num_skopt_call+1))
        print(p)

        # Train som
        som = run_som(features, size_x, xize_y, niter = int(p['niter']), sigma=p['sigma'], learning_rate=p['learning_rate'], 
              pca=pca, random_seed = random_seed)
        
        q_error = som.quantization_error(features)
        t_error = som.topographic_error(features)
        errors.append((num_skopt_call, q_error, t_error))
        
        def get_score(q_error, t_error):
            
            return math.sqrt(q_error * q_error + t_error * t_error)
        
        score = get_score(q_error, t_error)
        print( q_error )
        print( t_error )
        print( score )

        num_skopt_call += 1

        return score

    search_result = gp_minimize( func = fitness, dimensions = dimensions,
                                 acq_func = 'EI', # Expected Improvement
                                 n_calls = num_calls, x0 = prior_values )

    params = pd.DataFrame(search_result['x_iters'])
    params.columns = [*prior_names]
    params = params.rename_axis('call').reset_index()
    scores = pd.DataFrame(search_result['func_vals'])
    scores.columns = ['score']
    result = pd.concat([params, scores], axis=1)
    result = result.sort_values(by=['score'])
    errors_frame = pd.DataFrame(errors, columns = ['call', 'q_error', 't_error'])
    result = pd.merge(result, errors_frame, on=['call'])   
    
    return result


def run_tsne(features, perplexity = 40, plot=True):
    
    #pca_50 = PCA(n_components=50)
    # low early_ex looks good
    #pca_result_50 = pca_50.fit_transform(list(features))
    tsne_model = TSNE(perplexity=perplexity, #early_exaggeration = 500, # learning_rate = 500.00, early_exaggeration = 50,
                      n_components=2, init='pca', n_iter=1000, random_state=23, verbose=2)

    tsne_values = tsne_model.fit_transform(list(features))
    x = tsne_values[:,0]
    y = tsne_values[:,1]
    
    if plot == True:
        
        plt.figure(figsize=(10,10))
        plt.scatter(x,y, s=1)
        plt.show()
    
    return x,y

    
def run_kmeans(features, n_clusters=4):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    preds = kmeans.fit_predict(features)
    return preds, kmeans.cluster_centers_


def elbow(max_clusters, features):
    
    wcss = []
    for i in range(1, max_clusters):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(features)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_clusters), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def kmeans_som(som, features, size_x, size_y, n_clusters=4, run_elbow = False, max_cls=10):
    
    weights = som.get_weights()
    weights = weights.reshape(size_x * size_y, weights.shape[2])
    
    if run_elbow == False:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        pred_weights = kmeans.fit_predict(weights)

        weights = weights.reshape(size_x , size_y, weights.shape[1])
        pred_weights = pred_weights.reshape(size_x , size_y,)

        #colors = 'rgbcy'
        if n_clusters < 8:
            colors = 'rgbcymk'
        else:
            colors = []
            for i in range(n_clusters):
                colors.append('#%06X' % randint(0, 0xFFFFFF))
            
        win_map = som.win_map(features)
        plt.figure(figsize=(14, 14))
        the_grid = GridSpec(size_x, size_y)

        for i in range(size_x):
            for j in range(size_y):

                ax = plt.subplot(the_grid[i, j])
                c = pred_weights[i,j]
                ax.set_facecolor(colors[c])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

                if (i, j) in win_map.keys():
                    plt.plot(np.mean(win_map[i, j], axis=0))

        plt.show()
        
        return pred_weights
        
    else:
        elbow(max_cls, weights)
    
    
def som_to_kmeans_cluster(df, preds):
    
    def to_kmeans( cluster ):
        kmeans_cluster = preds[cluster]
        return kmeans_cluster
    
    df['cluster_som_kmeans'] = df['cluster'].apply(to_kmeans)
    
    
    
def plot_kmeans_tsne(df, col, n_clusters, plot_avg = True):
    
    if n_clusters < 8:
        colors = 'rgbcymk'
    else:
        colors = []
        for i in range(n_clusters):
            colors.append('#%06X' % randint(0, 0xFFFFFF))

    plt.figure(figsize=(10,10))
    for i in range(n_clusters):

        cls = df[df[col]==i]
        plt.scatter(cls['tsne_x'], cls['tsne_y'], c = colors[i], s = 1)
    
    plt.show()
    
    if plot_avg == True:
        for i in range(n_clusters):

            cls = df[df[col]==i]
            print(colors[i])
            plt.figure(figsize=(3,2))
            for vec in cls['waveform']:
                plt.plot(vec, color='gray', alpha=.01)
            plt.plot(np.mean(cls['waveform'], axis=0))
            plt.show()
    
    
    