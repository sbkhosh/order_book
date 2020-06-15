#!/usr/bin/python3

import base64
import clusterlib
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import datetime
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import scipy
import scipy.cluster.hierarchy as hac
import statsmodels.api as sm
import sys, os
import yaml

from dash.dependencies import Output, Input, State
from dt_read import DataProcessor
from plotly.subplots import make_subplots
from scipy import stats
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import pdist
from statsmodels.sandbox.regression.predstd import wls_prediction_std

color_1 = '#87189D'
STYLE_1 = {'font-family': 'Calibri','font-size':50,'width': '33%','display':'inline-block'}
STYLE_2 = {'font-family': 'Calibri','font-size':15,'width': '50%','display':'inline-block'}
STYLE_3 = {'width': '25%', 'float': 'left', 'display': 'inline-block'}
STYLE_4 = {'height': '100%', 'width': '100%'}
STYLE_5 = {'font-family': 'Calibri', 'color': color_1}
STYLE_6 = {'color': color_1}
STYLE_7 = {'height': '50%', 'width': '100%'}

def nav_menu():
    nav = dbc.Nav(
        [
            dbc.NavLink("Cluster", href='/page-1', id='page-1-link', style=STYLE_1),
        ],
        pills=True
        )
    return(nav)

def df_to_table(df):
    return(dbc.Table.from_dataframe(df,
                                    bordered=True,
                                    dark=True,
                                    hover=True,
                                    responsive=True,
                                    striped=True))

def get_data_all():
    obj_reader = DataProcessor('data_in','data_out','conf_model.yml')
    obj_reader.read_prm()
    obj_reader.process()
    return(obj_reader.trade_data,obj_reader.bid_data,obj_reader.ask_data,obj_reader.volume_data,
           obj_reader.df_trade_cluster,obj_reader.df_bid_cluster,obj_reader.df_ask_cluster)

def get_conf_helper():
    obj_helper = DataProcessor('data_in','data_out','conf_help.yml')
    obj_helper.read_prm()
    cut_cluster = obj_helper.conf.get('cut_cluster')
    cut_cluster_num = obj_helper.conf.get('cut_cluster_num')
    max_cluster_rep = obj_helper.conf.get('max_cluster_rep')
    return(cut_cluster,cut_cluster_num,max_cluster_rep)

####################################################################################################################################################################################
#                                                                                raw data and parameters                                                                           # 
####################################################################################################################################################################################
trade_data, bid_data, ask_data, volume_data, df_trade_cluster, df_bid_cluster, df_ask_cluster = get_data_all()
cut_cluster, cut_cluster_num, max_cluster_rep = get_conf_helper()
options_max_cluster = [{'label': i, 'value': i} for i in range(int(max_cluster_rep))]
####################################################################################################################################################################################
#                                                                                                                                                                                  # 
####################################################################################################################################################################################

def get_layout(idx_page):
    html_res = \
    html.Div([
        html.Div([
            html.Div(html.H6('Category'),style=STYLE_6),
            dcc.Dropdown(
                id='category-dropdown-'+str(idx_page),
                options=[{'label': i, 'value': i} for i in ['Trade','Bid','Ask']],
                value='Trade',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            html.Div(html.H6('Cluster Method'),style=STYLE_6),
            dcc.Dropdown(
                id='method-dropdown-'+str(idx_page),
                options=[{'label': i, 'value': i} for i in ['single','complete','average','weighted','centroid','median','ward']],
                value='ward',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            html.Div(html.H6('Cluster Metric'),style=STYLE_6),
            dcc.Dropdown(
                id='metric-dropdown-'+str(idx_page),
                options=[{'label': i, 'value': i} for i in ['euclidean','correlation','cosine','dtw']],
                value='euclidean',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            html.Div(html.H6('Cluster max #'),style=STYLE_6),
            dcc.Dropdown(
                id='max-cluster-dropdown-'+str(idx_page),
                options=[{'label': i, 'value': i} for i in range(int(max_cluster_rep))],
                value='12',
                style=STYLE_2
            )
            ],style=STYLE_3),
        html.Div([
            html.Div(html.P([html.Br(),html.H2(html.B('Cluster dendrogram - Clustered time series')),html.Br()]), style=STYLE_5),
            html.Img(id = 'cluster-plot-'+str(idx_page),
                           src = '',
                           style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.H6('Cluster Selected (only select those with cluster_size larger than one for the DTW analysis)'),style=STYLE_6),
            dcc.Dropdown(
                id='selected-cluster-dropdown-'+str(idx_page),
                value='9',
                style=STYLE_2
            )
        ]),
        html.Div(
            id='cluster-table-'+str(idx_page),
            className='tableDiv'
        ),
        html.Div([
            html.Div(html.P([html.Br(),html.H2(html.B('Dynamic time warping distance between pairs from selected cluster. The closer this distance is to 0, the more similar are the pairs'))]), style=STYLE_5),
            html.Img(id = 'dtws-uniq-plot-'+str(idx_page),
                           src = '',
                           style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.P([html.Br(),html.H2(html.B('Reccurence plot'))]), style=STYLE_5),
            html.Img(id = 'rec-plot-'+str(idx_page),
                           src = '',
                           style=STYLE_4)
        ]),
        html.Div([
            html.Div(html.P([html.Br(),html.H2(html.B('Matrix plot'))]), style=STYLE_5),
            html.Img(id = 'mtrx-plot-'+str(idx_page),
                           src = '',
                           style=STYLE_4)
        ]),
    ])
    return(html_res)

def cluster_draw(df_all, method, metric, max_cluster, selected_cluster, idx_page, df_base, category, ts_space=5):
    df_res, Z, ddata, dm = clusterlib.maxclust_draw_rep(df_all.iloc[:,:], method, metric, int(max_cluster), idx_page, 5)
    
    filename_0 = 'data_out/max_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(max_cluster)+'_idx_'+str(idx_page)
    image_name_0 = filename_0+".png"
    location_0 = os.getcwd() + '/' + image_name_0
    with open('%s' %location_0, "rb") as image_file_0:
        encoded_string_0 = base64.b64encode(image_file_0.read()).decode()
    encoded_image_0 = "data:image/png;base64," + encoded_string_0

    clusterlib.get_dtw_uniq_cluster(df_res, df_all, method, metric, max_cluster, selected_cluster, idx_page)
    filename_5 = 'data_out/dtw_uniq_cluster_draw_'+str(method)+'_'+str(metric)+'_'+str(selected_cluster)+'_idx_'+str(idx_page)
    image_name_5=filename_5+".png"
    location_5 = os.getcwd() + '/' + image_name_5
    with open('%s' %location_5, "rb") as image_file_5:
        encoded_string_5 = base64.b64encode(image_file_5.read()).decode()
    encoded_image_5 = "data:image/png;base64," + encoded_string_5

    clusterlib.rec_plot(df_base,category,idx_page)
    filename_6 = 'data_out/rec_plot_'+str(category.lower())+'_idx_'+str(idx_page)
    image_name_6=filename_6+".png"
    location_6 = os.getcwd() + '/' + image_name_6
    with open('%s' %location_6, "rb") as image_file_6:
        encoded_string_6 = base64.b64encode(image_file_6.read()).decode()
    encoded_image_6 = "data:image/png;base64," + encoded_string_6

    clusterlib.matrix_plot(df_all,category,idx_page)
    filename_7 = 'data_out/matrix_plot_'+str(category.lower())+'_idx_'+str(idx_page)
    image_name_7=filename_7+".png"
    location_7 = os.getcwd() + '/' + image_name_7
    with open('%s' %location_7, "rb") as image_file_7:
        encoded_string_7 = base64.b64encode(image_file_7.read()).decode()
    encoded_image_7 = "data:image/png;base64," + encoded_string_7

    return(encoded_image_0,df_res,encoded_image_5,encoded_image_6,encoded_image_7)
    
###################
# core of the app #  
###################
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP],meta_tags=[{"content": "width=device-width"}])
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
    html.Div([
        html.H1(nav_menu())]),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),    
    ],                     
)

@app.callback(
    [Output(f"page-{i}-link", "active") for i in range(1, 4)],
    [Input("url", "pathname")],
)
def toggle_active_links(pathname):
    if pathname == "/":
        return True, False, False
    return [pathname == f"/page-{i}" for i in range(1, 4)]
    
@app.callback(
    Output('selected-cluster-dropdown-1', 'options'),
    [Input('max-cluster-dropdown-1', 'value')]
)
def update_cluster_dropdown(max_cluster_val):
    options=[{'label': opt, 'value': opt} for opt in range(1,int(max_cluster_val)+1)]
    return(options)

@app.callback(
    Output('metric-dropdown-1', 'options'),
    [Input('method-dropdown-1', 'value')]
)
def update_cluster_dropdown(method_dropdown_val):
    if(method_dropdown_val == 'centroid' or method_dropdown_val == 'median' or method_dropdown_val == 'ward'):
        options=[{'label': 'euclidean', 'value':'euclidean'}]
    else:
        options=[{'label': opt, 'value': opt} for opt in ['euclidean','correlation','cosine','dtw']]
    return(options)

##########################################################################################################################################################################################
#                                                                                        page_1
##########################################################################################################################################################################################
page_1_layout = html.Div([ get_layout(1) ])

@app.callback([Output('cluster-plot-1', 'src'),
               Output('cluster-table-1', 'children'),
               Output('dtws-uniq-plot-1', 'src'),
               Output('rec-plot-1', 'src'),
               Output('mtrx-plot-1', 'src')
               ],
              [Input("category-dropdown-1", "value"),
               Input("method-dropdown-1", "value"),
               Input("metric-dropdown-1", "value"),
               Input("max-cluster-dropdown-1", "value"),
               Input("selected-cluster-dropdown-1", "value")
              ]
)
def update_fig(category,method,metric,max_cluster,selected_cluster):
    max_cluster = int(max_cluster)
    if(category == 'Trade'):
        df_all = df_trade_cluster
        df_base = trade_data
    elif(category == 'Bid'):
        df_all = df_bid_cluster
        df_base = bid_data
    elif(category == 'Ask'):
        df_all = df_ask_cluster
        df_base = ask_data

    encoded_image_0, df_res, encoded_image_5, encoded_image_6, encoded_image_7 = cluster_draw(df_all, method, metric, max_cluster, selected_cluster, 1, df_base, category, 5)
    cluster_html_table = df_to_table(df_res)                                                                  
    return(encoded_image_0, cluster_html_table, encoded_image_5, encoded_image_6, encoded_image_7)
   
####################################################################################################################################################################################
#                                                                                            page display                                                                          # 
####################################################################################################################################################################################
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    
if __name__ == '__main__':
    app.run_server(debug=True)
