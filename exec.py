#!/usr/bin/python3

import pandas as pd
import matplotlib
import warnings

from dt_help import Helper
from dt_model import OrderBook
from dt_read import DataProcessor
from pandas.plotting import register_matplotlib_converters

warnings.filterwarnings('ignore',category=FutureWarning)
pd.options.mode.chained_assignment = None 
register_matplotlib_converters()

if __name__ == '__main__':
    obj_helper = Helper('data_in','conf_help.yml')
    obj_helper.read_prm()
    
    fontsize = obj_helper.conf['font_size']
    matplotlib.rcParams['axes.labelsize'] = fontsize
    matplotlib.rcParams['xtick.labelsize'] = fontsize
    matplotlib.rcParams['ytick.labelsize'] = fontsize
    matplotlib.rcParams['legend.fontsize'] = fontsize
    matplotlib.rcParams['axes.titlesize'] = fontsize
    matplotlib.rcParams['text.color'] = 'k'
    
    obj_0 = DataProcessor('data_in','data_out','conf_model.yml')
    obj_0.read_prm()
    obj_0.process()

    model = OrderBook(obj_0.trade_data,obj_0.bid_data,obj_0.ask_data,obj_0.volume_data,obj_0.all_common_times)
    model.get_pattern()
    
