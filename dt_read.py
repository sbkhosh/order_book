#!/usr/bin/python3

import numpy as np
import os
import pandas as pd
import string
import yaml

from dt_help import Helper
from yahoofinancials import YahooFinancials

class DataProcessor():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {}, input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))
        
    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)

    @Helper.timing
    def process(self):
        file_in = self.conf.get('file_in')
        use_cols = self.conf.get('use_cols')

        self.data = pd.read_excel('/'.join((self.input_directory,file_in)),usecols=use_cols,header=None,skiprows=[0,1,2,3,4])
       
        # mapping columns numbers with the selected xlsx file letters 
        dct = dict(enumerate(string.ascii_uppercase))
        dct = {value:key for key, value in dct.items()}
        splits = use_cols.split(',')
        
        headers_num = []
        for el in splits:
            splt = el.split(':')
            lst = list(map(chr, range(ord(splt[0]),ord(splt[1])+1)))
            headers_num.append([dct[x] for x in lst])

        cols_price = [ 'Dates', 'Type', 'Price', 'Size' ]
        cols_volume = [ 'Dates', 'Volume', 'Cumulative Volume' ]

        self.trade_data = self.data[headers_num[0]].dropna()
        self.bid_data = self.data[headers_num[1]].dropna()
        self.ask_data = self.data[headers_num[2]].dropna()
        self.volume_data = self.data[headers_num[3]].dropna()

        self.trade_data.columns = cols_price
        self.bid_data.columns = cols_price
        self.ask_data.columns = cols_price
        self.volume_data.columns = cols_volume

        self.trade_data.drop(self.trade_data.index[0],inplace=True)
        self.bid_data.drop(self.bid_data.index[0],inplace=True)
        self.ask_data.drop(self.ask_data.index[0],inplace=True)
        self.volume_data.drop(self.volume_data.index[0],inplace=True)
       
        self.trade_data.drop(columns=['Type'],inplace=True)
        self.bid_data.drop(columns=['Type'],inplace=True)
        self.ask_data.drop(columns=['Type'],inplace=True)
       
        self.trade_data[['Price', 'Size']].astype(float)
        self.ask_data[['Price', 'Size']].astype(float)
        self.bid_data[['Price', 'Size']].astype(float)
        self.volume_data[['Volume', 'Cumulative Volume']].astype(float)

        self.trade_data['Dates'] = self.trade_data['Dates'].dt.time
        self.bid_data['Dates'] = self.bid_data['Dates'].dt.time
        self.ask_data['Dates'] = self.ask_data['Dates'].dt.time
        self.volume_data['Dates'] = self.volume_data['Dates'].dt.time
        
        self.trade_data.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        self.ask_data.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        self.bid_data.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)
        self.volume_data.replace([np.inf, -np.inf, 'inf', '-inf'], np.nan).dropna(inplace=True)

        self.trade_data['Dates_wo_sec'] = self.trade_data['Dates'].apply(lambda x: "{:02d}"':'"{:02d}".format(x.hour,x.minute))
        self.bid_data['Dates_wo_sec'] = self.bid_data['Dates'].apply(lambda x: "{:02d}"':'"{:02d}".format(x.hour,x.minute))
        self.ask_data['Dates_wo_sec'] = self.ask_data['Dates'].apply(lambda x: "{:02d}"':'"{:02d}".format(x.hour,x.minute))
        self.volume_data['Dates_wo_sec'] = self.volume_data['Dates'].apply(lambda x: "{:02d}"':'"{:02d}".format(x.hour,x.minute))
        
        days_trade = self.trade_data['Dates_wo_sec']
        days_bid = self.bid_data['Dates_wo_sec']
        days_ask = self.ask_data['Dates_wo_sec']
        days_volume = self.volume_data['Dates_wo_sec']

        self.all_common_times = sorted(list(set.intersection(*map(set, [days_trade,days_bid,days_ask,days_volume]))))
        
        self.trade_data.set_index(['Dates'],inplace=True)
        self.bid_data.set_index(['Dates'],inplace=True)    
        self.ask_data.set_index(['Dates'],inplace=True)
        self.volume_data.set_index(['Dates'],inplace=True)
        self.volume_data['Cumulative Volume'].iloc[0] = self.volume_data['Volume'].iloc[0]

        # preparing data for clustering
        dct_trade = {};dct_bid = {};dct_ask = {}
        trade_dates = sorted(list(set(self.trade_data.index)))
        trade_split = [ self.trade_data[self.trade_data.index==el] for el in trade_dates ]
        for i,el in enumerate(list(map(str,trade_dates))):
            dct_trade[el] = {'sizes': list(trade_split[i]['Size'].values),
                             'prices': list(trade_split[i]['Price'].values)}

        bid_dates = sorted(list(set(self.bid_data.index)))
        bid_split = [ self.bid_data[self.bid_data.index==el] for el in bid_dates ]
        for i,el in enumerate(list(map(str,bid_dates))):
            dct_bid[el] = {'sizes': list(bid_split[i]['Size'].values),
                           'prices': list(bid_split[i]['Price'].values)}

        ask_dates = sorted(list(set(self.ask_data.index)))
        ask_split = [ self.ask_data[self.ask_data.index==el] for el in ask_dates ]
        for i,el in enumerate(list(map(str,ask_dates))):
            dct_ask[el] = {'sizes': list(ask_split[i]['Size'].values),
                           'prices': list(ask_split[i]['Price'].values)}

        res_trade = {k: v['sizes'] for k,v in dct_trade.items()}
        res_bid = {k: v['sizes'] for k,v in dct_bid.items()}
        res_ask = {k: v['sizes'] for k,v in dct_ask.items()}

        self.df_trade_cluster = pd.DataFrame(data=res_trade.values(),index=sorted(list(set(self.trade_data.index)))).fillna(0)
        self.df_bid_cluster = pd.DataFrame(data=res_bid.values(),index=sorted(list(set(self.bid_data.index)))).fillna(0)
        self.df_ask_cluster = pd.DataFrame(data=res_ask.values(),index=sorted(list(set(self.ask_data.index)))).fillna(0)
        
    def write_to(self,name,flag):
        filename = os.path.join(self.output_directory,name)
        try:
            if('csv' in flag):
                self.data.to_csv(str(name)+'.csv')
            elif('xls' in flag):
                self.data.to_excel(str(name)+'xls')
        except:
            raise ValueError("not supported format")

