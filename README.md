# StromboliWF
Unsupervised analysis of the time evolution of clusters of VLP signal waveforms (timeseries) from the Stromboli volcano. 

## Cutting of the signals
The selected signals can be cutted with [day_to_ascii.py](https://github.com/llayer/StromboliWF/blob/master/day_to_ascii.py).

## Feature extraction
Filtering, decimating, LPC and amplitude parametrization is done with the script [ascii_to_param.py](https://github.com/llayer/StromboliWF/blob/master/ascii_to_param.py). A visualization for an exemplary signal can be found in [parametrization.ipynb](https://github.com/llayer/StromboliWF/blob/master/parametrization.ipynb).

## Autoencoders
The script [autoencoder.ipynb](https://github.com/llayer/StromboliWF/blob/master/autoencoder.ipynb) implements autoencoders and can be run on Google Colab.

## t-SNE and SOM optimization
The expensive t-SNE algorithm and the Bayesian optimization of the SOM can be run with [tsne_somopt.ipynb](https://github.com/llayer/StromboliWF/blob/master/tsne_somopt.ipynb) and then saved for later usage.

## Clustering with K-Means
An example to cluster the filtered signals can be found in [cluster.ipynb](https://github.com/llayer/StromboliWF/blob/master/cluster.ipynb). The SOM and clustering with K-Means on the weight vectors of the SOM is done in [SOM.ipynb](https://github.com/llayer/StromboliWF/blob/master/SOM.ipynb)

## Comparison of the time evolution
The time evolution of the clusters of the different methods can be found in [ratio.ipynb](https://github.com/llayer/StromboliWF/blob/master/ratio.ipynb)
