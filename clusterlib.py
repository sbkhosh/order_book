import base64
import chart_studio.plotly as py
import datetime as dt
import itertools
import joblib
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import plotly.offline as pyoff
import plotly.graph_objs as go
import re
import scipy
import scipy.cluster.hierarchy as hac
import scipy.spatial.distance as ssd
import scipy.stats as stats
import seaborn as sns
import sklearn
import statsmodels.api as sm
import sys, os
import time
import yaml
import warnings

from array import array
from collections import Counter
from datetime import datetime
from dtaidistance import dtw
from fastdtw import fastdtw
from heapq import nlargest
from joblib import Parallel, delayed
from math import exp, log, sqrt
from matplotlib import style
from numpy.random import rand
from pandas import DataFrame
from pandas.plotting import register_matplotlib_converters,scatter_matrix
from pandas.tseries.offsets import BDay
from pprint import pprint
from pylab import *
from random import random
from scipy.stats import norm,shapiro
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy.stats import trim_mean
from sklearn import cluster
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.manifold import MDS
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 

fontsize = 20

matplotlib.rcParams['axes.labelsize'] =  fontsize
matplotlib.rcParams['xtick.labelsize'] = fontsize-2
matplotlib.rcParams['ytick.labelsize'] = fontsize-2
matplotlib.rcParams['legend.fontsize'] = fontsize
matplotlib.rcParams['axes.titlesize'] = fontsize
matplotlib.rcParams['text.color'] = 'k'

def format_output(df):
    df.columns = [''] * len(df.columns)
    df = df.to_string(index=False)
    return(df)
    
def KL(P,Q):
    """ Epsilon is used here to avoid conditional code for
    checking that neither P nor Q is equal to 0. """
    epsilon = 0.00001
    
    P = P+epsilon
    Q = Q+epsilon
    
    divergence = np.sum(P*np.log(P/Q))
    return(divergence)
 
def dtw_ts_fast(ts1,ts2):
    d, path = fastdtw(ts1,ts2)
    return(d)

def get_dtws_cluster(data, method, metric, max_cluster, cluster_select, idx_page):
    cwd = os.getcwd()
    path = cwd+'/dtws_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pkl'
    if(os.path.exists(path)):
        res = pd.read_pickle(path)
    else:
        cols = data.columns.values
        combs = list(itertools.permutations(cols,2))
        dtws = Parallel(n_jobs=-1)(delayed(dtw_ts_fast)(data[str(el[0])],data[str(el[1])]) for el in combs)
        res = pd.DataFrame()
        res['1st'] = [ str(combs[k][0]) for k,el in enumerate(combs) ]
        res['2nd'] = [ str(combs[k][1]) for k,el in enumerate(combs) ]
        res['dtw'] = [ dtws[k] for k,el in enumerate(combs) ]
        max_dtw = max(res['dtw'])
        res['dtw'] = [ el/max_dtw for el in res['dtw'] ]
        res.to_pickle(path)
    return(res)

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = hac.dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def print_clusters(timeSeries,Z,k,cols_raw,method,metric,plot=False):
    cols = cols_raw
    # k Number of clusters I'd like to extract
    results = fcluster(Z, k, criterion='maxclust')

    # check the results
    s = pd.Series(results)
    clusters = s.unique()
    
    for count,c in enumerate(clusters):
        cluster_indices = s[s==c].index
        print("Cluster %d number of entries %d" % (c, len(cluster_indices)))
        if plot:
            timeSeries.T.iloc[:,cluster_indices].plot(figsize=(32,20))
            plt.savefig('data_out/selected_'+str(count)+'_cluster_'+str(method)+'_'+str(metric)+'.pdf')
            plt.show()
            # plt.legend(loc='center left',bbox_to_anchor=(1.0,0.5), fancybox=True, shadow=True)

def elbow_method(Z,method_select,metric_select,idx_select,plotting):
    last = Z[:idx_select,2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)

    acceleration = np.diff(last, 2)  # 2nd derivative of the distances
    acceleration_rev = acceleration[::-1]
    k = acceleration_rev.argmax() + 2  # if idx 0 is the max of this we want 2 clusters
    
    if(plotting):
        plt.figure(figsize=(32,20))
        plt.plot(idxs, last_rev)
        plt.plot(idxs[:-2] + 1, acceleration_rev)
        plt.title('last ' + str(idx_select) + ' points of sample - ' + str(method_select) + ' - ' + str(metric_select))
        plt.savefig('data_out/elbow_cluster_'+str(method_select)+'_'+str(metric_select)+'.pdf')
    return(k)

def fix_verts(ax, orient=1):
    for coll in ax.collections:
        for pth in coll.get_paths():
            vert = pth.vertices
            vert[1:3,orient] = scipy.average(vert[1:3,orient]) 
    
def clustering_matrix_plot(df,method_select,metric_select):
    df_raw = df[df.columns]
    # cg = sns.clustermap(X, metric="")
    # plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
    W = df_raw.values

    corr_matrix = df_raw.corr()
    corr_matrix = np.triu(corr_matrix)
    
    col_labels = df.columns.values
    row_labels = df.columns.values   
    h,labels_map = heatmapcluster(corr_matrix,
                       row_labels,
                       col_labels,
                       num_row_clusters=10,
                       num_col_clusters=10,
                       label_fontsize=6,
                       xlabel_rotation=90,
                       cmap=plt.cm.coolwarm,
                       show_colorbar=True,
                       top_dendrogram=True,
                       row_linkage=lambda x: hac.linkage(x.T, method=method_select,
                                                     metric=metric_select),
                       col_linkage=lambda x: hac.linkage(x.T, method=method_select,
                                                     metric=metric_select),
                       colorbar_pad = 1.5,
                       histogram=False)

    df_labels = pd.DataFrame.from_dict({'labels': labels_map})
    plt.savefig('data_out/matrix_cluster_'+str(method_select)+'_'+str(metric_select)+'.pdf')
    df_labels.to_csv('data_out/matrix_cluster_'+str(method_select)+'_'+str(metric_select)+'.csv')
    df_labels.to_excel('data_out/matrix_cluster_'+str(method_select)+'_'+str(metric_select)+'.xlsx')

def get_dtw_matrix(df):
    df_raw = df[df.columns]
    perms = list(permutations(df.columns.values,2))
    res = pd.DataFrame.from_items([(str(el[0]), fastdtw(df_raw[el[0]],df_raw[el[1]])[0]) for el in perms], orient='index',columns=[k[0] for k in perms])
    
def clustering_kmeans(df):
    df_raw = df[df.columns].T
    X = df_raw.values
    
    wcss = []
    K = range(1,100)
    for k in K:
        kmeans = KMeans(n_clusters=k,init='k-means++', random_state=426135).fit(X)
        print(kmeans.labels_)
        wcss.append(kmeans.inertia_)

    fig = plt.figure(figsize=(32,20))
    plt.figure(figsize=(32,20))
    plt.plot(K, wcss, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distortion')
    plt.title('elbow Method showing the optimal k')
    plt.show()
  
def add_distance(ddata, dist_threshold=None, fontsize=8):
    '''
    Plot cluster points & distance labels in dendrogram
    ddata: scipy dendrogram output
    dist_threshold: distance threshold where label will be drawn, if None, 1/10 from base leafs will not be labelled to prevent clutter
    fontsize: size of distance labels
    '''
    if dist_threshold==None:
        # add labels except for 1/10 from base leaf nodes
        dist_threshold = max([a for i in ddata['dcoord'] for a in i])/10
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
        y = sum(i[1:3])/2
        x = d[1]
        # only label above distance threshold
        if x > dist_threshold:
            plt.plot(x, y, 'o', c=c, markeredgewidth=0)
            plt.annotate(int(x), (x, y), xytext=(15, 3),
                        textcoords='offset points',
                        va='top', ha='center', fontsize=fontsize)

def maxclust_draw(df, method, metric, max_cluster, ts_space=1):
    '''
    Draw agglomerative clustering dendrogram based on maximum cluster criteron

    df: dataframe or arrays of timeseries
    method: agglomerative clustering linkage method
    metric: distance metrics
    max_cluster: maximum cluster size to flatten cluster
    ts_space: horizontal space for timeseries graph to be plotted

    gets the dendrogram with timeseries graphs on the side
    '''
    cols = df.T.columns
    # define gridspec space
    gs = gridspec.GridSpec(max_cluster,max_cluster)
    
    # add dendrogram to gridspec
    fig, ax = plt.subplots(figsize=(32,20))
    plt.subplot(gs[:, 0:max_cluster-ts_space])
    plt.xlabel('distance')
    plt.ylabel('cluster')
    
    # agglomerative clustering
    Z = hac.linkage(df, method=method, metric=metric)
    ddata = hac.dendrogram(Z, orientation='left',
                           truncate_mode='lastp',
                           p=max_cluster,
                           show_leaf_counts=True,
                           labels=cols,
                           show_contracted=True)

    # check with elbow method
    kcluster = elbow_method(Z,method,metric,30,False)
    coph_score,coph_dists = cophenet(Z,pdist(df))
    coph_score = round(coph_score,3)
    
    # add distance labels in dendrogram
    add_distance(ddata)

    # get cluster labels
    y = fcluster(Z, max_cluster, criterion='maxclust')
    y = pd.DataFrame(y,columns=['y'])
   
    # get individual names for each cluster
    df_clst = pd.DataFrame()
    df_clst['index']  = df.index
    df_clst['label']  = y

    # summarize info for output
    dct_sum = {'cluster': [], 'cluster_size': [], 'components': [], 'cophenet': [], 'cluster_elbow': []}
    for i in range(max_cluster):
        elements = df_clst[df_clst['label']==i+1]['index'].tolist()  
        size = len(elements)
        dct_sum['cluster'].append(i+1)
        dct_sum['cluster_size'].append(size)
        dct_sum['components'].append(elements)
        # dct_sum['cophenet'].append(coph_score)
        # dct_sum['cluster_elbow'].append(kcluster)
        # print('cluster {}: N = {}  {}'.format(i+1, size, elements))
    df_sum = pd.DataFrame.from_dict(dct_sum)
    
    # merge with original dataset
    dx=pd.concat([df.reset_index(drop=True), y],axis=1)

    # add timeseries graphs to gridspec
    for cluster in range(1,max_cluster+1):
        reverse_plot = max_cluster+1-cluster
        plt.subplot(gs[reverse_plot-1:reverse_plot,max_cluster-ts_space:max_cluster])
        plt.axis('off')
        for i in range(len(dx[dx['y']==cluster])):
            plt.plot(dx[dx['y']==cluster].T[:-1].iloc[:,i]);
    plt.tight_layout()

    plt.savefig('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.pdf')
    plt.savefig('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.png')
    df_sum.to_csv('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.csv')
    df_sum.to_excel('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'.xlsx')

def maxclust_draw_rep(df, method, metric, max_cluster, idx_page, ts_space=1):
    cols = df.T.columns
    # define gridspec space
    gs = gridspec.GridSpec(max_cluster,max_cluster)
    
    # add dendrogram to gridspec
    fig, ax = plt.subplots(figsize=(32,16))
    plt.subplot(gs[:, 0:max_cluster-ts_space])
    plt.xlabel('distance')
    plt.ylabel('cluster')

    # agglomerative clustering
    if(metric != 'dtw'):
        # Z = hac.linkage(df, method=method, metric=metric)
        dm = pdist(df.values,metric=metric)
        Z = hac.linkage(dm,method=method)
    elif(metric == 'dtw'):
        dm = pdist(df,lambda u,v: dtw_ts_fast(u,v))
        Z = hac.linkage(dm,method=method)

    ddata = hac.dendrogram(Z, orientation='left',
                           truncate_mode='lastp',
                           p=max_cluster,
                           show_leaf_counts=True,
                           labels=cols,
                           show_contracted=True)

    # check with elbow method
    kcluster = elbow_method(Z,method,metric,30,False)
    coph_score,coph_dists = cophenet(Z,pdist(df))
    coph_score = round(coph_score,3)
    
    # add distance labels in dendrogram
    add_distance(ddata)

    # get cluster labels
    y = fcluster(Z, max_cluster, criterion='maxclust')
    y = pd.DataFrame(y,columns=['y'])

    # get individual names for each cluster
    df_clst = pd.DataFrame()
    df_clst['index']  = df.index
    df_clst['label']  = y
    
    # summarize info for output
    dct_sum = {'cluster_idx': [], 'cluster_size': [], 'components': []}
    for i in range(max_cluster):
        elements = df_clst[df_clst['label']==i+1]['index'].tolist()
        size = len(elements)
        dct_sum['cluster_idx'].append(i+1)
        dct_sum['cluster_size'].append(size)
        dct_sum['components'].append('\n'+' | '.join(list(map(str,elements))))
    dct_sum['cluster_idx'] = dct_sum['cluster_idx'][::-1]
    df_sum = pd.DataFrame.from_dict(dct_sum)

    # # merge with original dataset
    # dx=pd.concat([df.reset_index(drop=True), y],axis=1)

    dgg = df.copy(deep=True)
    dgg['y']=y.values
    dgg['Dates']=[ str(el) for el in dgg.index ]
    cols_select = [ el for el in list(dgg.columns) if isinstance(el,int) ]

    df_plots = []
    for el in ddata['ivl']:
        if(type(el)==datetime.time):
            dtt = str(el)
            df_plots.append(list(dgg[dgg['Dates']==dtt][cols_select].values[0]))
        else:
            intt = int(el.split('(')[1].split(')')[0])
            ts_comps = df_sum[df_sum['cluster_size']==intt]['components'].values
            ts = [ re.sub(r"[^a-zA-Z0-9]\n", '', k).replace('\n','').replace(' ','') for k in ts_comps[0].split('|') ]
            df_plots.append([ list(dgg[dgg['Dates']==el][cols_select].values[0]) for el in ts ])
    
    for cluster in range(1,max_cluster+1): 
        plt.subplot(gs[max_cluster-cluster:max_cluster-cluster+1,max_cluster-ts_space:max_cluster])
        if(np.sum(list(any(isinstance(el, list) for el in df_plots[cluster-1]))) > 0):
            [ plt.plot(el) for el in df_plots[cluster-1] ]
        else:
            plt.plot(df_plots[cluster-1])
            
    plt.tight_layout()

    plt.savefig('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.png')
    df_sum.to_csv('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.csv')
    df_sum.to_excel('data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.xlsx')
    return(df_sum, Z, ddata, dm)
    
def rep_string(text):
    rep = {"\n": "", " | ": " "}
    rep = dict((re.escape(k), v) for k, v in rep.items()) 
    pattern = re.compile("|".join(rep.keys()))
    text = pattern.sub(lambda m: rep[re.escape(m.group(0))], text)
    return(text)

def get_qty_sum(df_res, df_raw, method, metric, max_cluster, idx_page):
    df_all_res = []
    Nclusters = len(df_res)
    assert Nclusters == int(max_cluster)

    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))
    for el in range(max_cluster):
        cmps = df_res['components'].iloc[el]
        res = rep_string(cmps).split(' ')
        df_compute = df_raw[res]
        for col_idx in df_compute.columns:
            df_compute['return_'+str(col_idx)] = df_compute[str(col_idx)].pct_change()
            df_compute.dropna(inplace=True)
            df_compute['cum_ret_'+str(col_idx)] = (1.0+df_compute['return_'+str(col_idx)]).cumprod() - 1.0
        df_all_res.append(df_compute)

    gs = gridspec.GridSpec(max_cluster,1)
    fig, ax = plt.subplots(figsize=(32,16),sharex=True)
    for el in range(max_cluster):
        reverse_plot = max_cluster-1-el
        plt.subplot(gs[reverse_plot,:])
        if(el != 0):
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
        df_select = df_all_res[el]
        plt.plot(df_select.index,df_select[[k for k in df_select.columns if 'cum_ret' in k]])
    # plt.tight_layout()

    plt.savefig('data_out/qty_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/qty_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)+'.png')

def get_qty_uniq_sum(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))

    df_all_res = df_raw[res]
    for col_idx in df_all_res.columns:
        df_all_res['return_'+str(col_idx)] = df_all_res[str(col_idx)].pct_change()
        df_all_res.dropna(inplace=True)
        df_all_res['cum_ret_'+str(col_idx)] = (1.0+df_all_res['return_'+str(col_idx)]).cumprod() - 1.0

    trace_all = []
    for el in [k for k in df_all_res.columns if 'cum_ret' in k]:
        trace_all.append(go.Scatter(x=df_all_res.index, y=df_all_res[el], line=dict(width=2), name=el.split('cum_ret_')[1]))
    
    data_all = trace_all
    layout_all = dict(title=go.layout.Title(text='', font=dict(size=18), x=0.5, xanchor='center'),
                      xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='Dates')),
                      yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='Values')),
                      paper_bgcolor='#fff',
                      plot_bgcolor='#fff',      
                      style={'font-family': 'Calibri','font-size':16})
    return({'data':data_all,'layout':layout_all})
        
def get_corr_uniq_cluster(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))
    df_all_res = df_raw[res]
    corr_matrix = df_all_res.corr()
    
    trace_heat = go.Heatmap(z=corr_matrix,
                            x=df_all_res.columns.values,
                            y=df_all_res.columns.values,
                            xgap=1, ygap=1,
                            colorbar_thickness=20,
                            colorbar_ticklen=3,
                            hoverinfo='text',
    )
       
    data_all = [trace_heat]
    layout_all = go.Layout(width=1920, height=1920,
                           xaxis_showgrid=False,
                           yaxis_showgrid=False,
                           yaxis_autorange='reversed',
                           paper_bgcolor='#fff',
                           plot_bgcolor='#fff',
                           
    )
    return({'data': data_all,'layout':layout_all})
   
def get_pairplot_uniq_cluster(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))
    df_all_res = df_raw[res]

    fig, ax = plt.subplots(figsize=(32,16))
    g = sns.PairGrid(df_all_res)
    g.map_diag(sns.kdeplot)
    g.map_offdiag(sns.kdeplot, n_levels=10);
    plt.tight_layout()

    plt.savefig('data_out/pairplot_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/pairplot_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def get_dtw_uniq_cluster(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))

    df_all_res = df_raw[res]
    
    res = get_dtws_cluster(df_all_res, method, metric, max_cluster, cluster_select, idx_page)
    res_htmap = res.pivot(index='1st', columns='2nd', values='dtw')
    fig, ax = plt.subplots(figsize=(32,16))
    sns.heatmap(res_htmap, annot = True, center=0, cmap='coolwarm', square=True)
    plt.tight_layout()

    plt.savefig('data_out/dtw_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/dtw_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def get_klb_uniq_cluster(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))

    df_all_res = df_raw[res]

    fig, ax = plt.subplots(figsize=(32,16))
    df_all_res.plot.kde(ax=ax)
    plt.tight_layout()
   
    plt.savefig('data_out/klb_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/klb_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def plot_clusters_hist(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page, Z):
    part_centroid = hac.fcluster(Z,max_cluster,criterion='maxclust')

    fig, ax = plt.subplots(figsize=(32,16))
    otpt = plt.hist(part_centroid[::-1],bins=max_cluster,ec='black')
    plt.xlabel('Cluster no.')
    plt.ylabel('Counts')
    plt.xticks(np.arange(max_cluster), [str(el) for el in np.arange(max_cluster)[::-1]])
    
    plt.savefig('data_out/clusters_hist_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/clusters_hist_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def plot_clusters_text(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page, dm):
    cols = df_raw.columns.values
    mds = MDS(2, dissimilarity='precomputed',n_jobs=-1)
    coords = mds.fit_transform(squareform(dm))
    x, y = coords[:, 0], coords[:, 1]

    fig, ax = plt.subplots(figsize=(32,16))
    plt.scatter(x,y)
    for (el, _x, _y) in zip(cols, x, y):
        plt.annotate(el, (_x, _y))
    plt.xticks([])
    plt.yticks([])

    plt.savefig('data_out/clusters_text_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/clusters_text_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def plot_clusters_uniq(df_res, df_raw, method, metric, max_cluster, cluster_select, idx_page, Z, ddata, dm):
    cmps = df_res['components'][int(max_cluster)-int(cluster_select)]
    res = rep_string(cmps).split(' ')
    df_raw = df_raw.T
    df_raw.columns = list(map(str,df_raw.columns))

    df_all_res = df_raw[res]        

    cols = df_all_res.columns.values
    combs = list(itertools.combinations(cols,2))

    fig, ax = plt.subplots(figsize=(32,16))
    for el in combs:
        plt.scatter(df_all_res[el[0]],df_all_res[el[1]],label=str(el[0])+'--'+str(el[1]))

    plt.legend()
    plt.savefig('data_out/clusters_uniq_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/clusters_uniq_draw_'+str(method)+'_'+str(metric)+'_'+str(cluster_select)+'_idx_'+str(idx_page)+'.png')

def rec_def(s, eps=0.1, steps=10):
    d = pdist(s[:,None])
    d = np.floor(d/eps)
    d[d>steps] = steps
    Z = squareform(d)
    return(Z)
    
def rec_plot(df,category,idx_page):
    dates_perf = df.index
    s = df['Size']
    epsilons = [0.10,0.30,0.60,0.80]
        
    a = [rec_def(s, eps=el) for el in epsilons]

    size = 0.33
    alignement = 0.1

    fig = plt.figure(figsize=(32,16))
    font = {
        'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 16,
        }

    plt.suptitle(str(category))
    ax_recurrence_0 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax_recurrence_0.clear()

    xmin,xmax = mdates.datestr2num([str(dates_perf[0]),str(dates_perf[-1])])
    ymin,ymax = mdates.datestr2num([str(dates_perf[0]),str(dates_perf[-1])])    

    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax_recurrence_0.xaxis.set_major_formatter(xfmt)
    ax_recurrence_0.yaxis.set_major_formatter(xfmt)
    
    # recurrence plots
    ax_recurrence_0.imshow(a[0], extent=[xmin,xmax,ymax,ymin], cmap='gray')
    ax_recurrence_0.set_title("Recurrence plot - epsilon = "+str(epsilons[0]), fontdict=font)
    ax_recurrence_0.xaxis_date()
    ax_recurrence_0.yaxis_date()

    # plt.tight_layout()

    plt.savefig('data_out/rec_plot_'+str(category.lower())+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/rec_plot_'+str(category.lower())+'_idx_'+str(idx_page)+'.png')

def matrix_plot(df,category,idx_page):
    dates_perf = df.index
        
    fig = plt.figure(figsize=(32,16))
    font = {
        'family': 'serif',
        'color':  'darkblue',
        'weight': 'normal',
        'size': 16,
        }

    plt.suptitle(str(category))
    ax_recurrence_0 = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    ax_recurrence_0.clear()

    xmin,xmax = mdates.datestr2num([str(dates_perf[0]),str(dates_perf[-1])])
    # ymin,ymax = mdates.datestr2num([str(dates_perf[0]),str(dates_perf[-1])])    

    xfmt = mdates.DateFormatter('%H:%M:%S')
    ax_recurrence_0.xaxis.set_major_formatter(xfmt)
    # ax_recurrence_0.yaxis.set_major_formatter(xfmt)

    # recurrence plots
    ax_recurrence_0.imshow(df.T.values, extent=[xmin, xmax, -1.5, len(df.columns)-1.5], cmap='gray')
    ax_recurrence_0.set_title("Heatmap of time evolution of trade orders", fontdict=font)
    ax_recurrence_0.xaxis_date()
    # ax_recurrence_0.yaxis_date()

    # plt.tight_layout()

    plt.savefig('data_out/matrix_plot_'+str(category.lower())+'_idx_'+str(idx_page)+'.pdf')
    plt.savefig('data_out/matrix_plot_'+str(category.lower())+'_idx_'+str(idx_page)+'.png')

    
