#coding : utf-8

import os
import time
import datetime
import pytz
import csv
import numpy as np
import pandas as pd
import oandapy
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingRegressor
from sklearn.externals import joblib
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.models import load_model
import copy
import warnings
from config import *
from collections import namedtuple
warnings.filterwarnings("ignore")

# 注文
# 引数 : 取引通貨, 買い 1 売り -1
# 戻値 : 注文ID
def order(api, acid, currency, bs, amount):
    if bs == 1:
        side = "buy"
    if bs == -1:
        side = "sell"
    response = api.create_order(account_id=acid, instrument=currency,
                                  units=amount, side=side, type="market")
    try:
        iden = int(response["tradeOpened"]["id"]) # id 取得
    except:
        iden = int(response["tradeOpened"][0]["id"])
    return iden

#注文決済
#引数：アカウントID 取引通貨 買い 1 売り -1 注文ID
#戻り値: なし
def closeOrder(api, acid, currency, iden):
    api.close_trade(account_id = acid, trade_id = int(iden))
    print(currency+":決済 "+str(iden))

#新規注文中止通知
#引数: なし
#戻り値: 取引終了時間  True 取引時間 False
def stopNewOrder():
    today = datetime.datetime.today()
    #print(today.weekday())
    #5:Saturday
    if  today.weekday() == stopOrderTime.weekday and\
        today.hour >= stopOrderTime.hour and\
        today.minute >= stopOrderTime.minute:
        return True
    else:
        return False

# 待機
# 引数 waitMin : 待機間隔(minute)
# 戻り値:なし
def waitMinute(waitMin):
    pre_min = datetime.datetime.today().minute//waitMin
    while True:
        now_min = datetime.datetime.today().minute//waitMin
        #print(pre_min, now_min)
        if pre_min != now_min:
            break
        time.sleep(5)
