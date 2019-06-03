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
from func import *
from collections import namedtuple
warnings.filterwarnings("ignore")
np.set_printoptions(threshold = np.inf) #指数表示の禁止

# OANDA 
ACCOUNT_ID = None
ACCESS_TOKEN = None
oanda = oandapy.API(environment = "live", access_token = ACCESS_TOKEN)

currency = {}
for cur in CURRENCY:
    currency[cur] = CurrencyManager(oanda, ACCOUNT_ID, cur, timezone, tradeTime,
                          threshold, spreadReference, ss, nn)
flagStopOrder = False


#取引開始時間
START_TRADE_HOUR = 14
START_TRADE_MINUTE = 50
#トレード開始
#引数: なし
#戻り値: True 取引開始時間  False 取引時間前
def start_trade():
    today = datetime.datetime.today()
    #print(today.weekday())
    #5:Saturday
    if  today.weekday() == 0 and today.hour >= START_TRADE_HOUR and today.minute >= START_TRADE_MINUTE:
        return True
    else:
        return False
#---------------------取引開始-----------------------#
if __name__ == "__main__":
    """while not start_trade():
        print("Wait")
        time.sleep(10)"""

    # 過去の価格をセット
    for cur in CURRENCY:
        currency[cur].setHistPrice()
        currency[cur].setTime()
    # 取引時間まで待つ
    waitMinute(5)
    #time.sleep(5)
    # 注文する取引通貨の判断
    tradeList = []
    for cur in CURRENCY:
        currency[cur].updataHistPrice() # 価格データの更新
        if currency[cur].isTradeTime(): # 取引ok?
            currency[cur].setInputData() # 入力データセット
            currency[cur].setOutput() # 出力値セット
            if currency[cur].canOrder():
                tradeList.append(cur)
    # 注文
    for cur in CURRENCY:
        if not cur in tradeList: # 注文しない通貨ペア
            orderHist[cur].append(0)
            continue
        amount = int(currency[cur].orderAmount / len(tradeList))
        iden = order(oanda, ACCOUNT_ID, cur, currency[cur].bs, amount)
        if currency[cur].bs == 1:
            orderIdBuy[cur].append(iden)
        if currency[cur].bs == -1:
            orderIdSell[cur].append(iden)
        orderHist[cur].append(currency[cur].bs)

    while True:
        waitMinute(5)
        #time.sleep(5)
        if not flagStopOrder and stopNewOrder(): # 注文取引時間の判断
            flagStopOrder = True

        # 注文する取引通貨の判断
        tradeList = []
        for cur in CURRENCY:
            if flagStopOrder: # 注文取引時間外の場合
                continue
            currency[cur].updataHistPrice() # 価格データの更新
            if currency[cur].isTradeTime(): # 取引ok?
                currency[cur].setInputData() # 入力データセット
                currency[cur].setOutput() # 出力値セット
                if currency[cur].canOrder():
                    if -currency[cur].bs in orderHist[cur]: # 両建て発生
                        currency[cur].flagCrossOrder = True
                        continue
                    tradeList.append(cur)

        # 両建て決済
        for cur in CURRENCY:
            if currency[cur].flagCrossOrder: # 両建ての場合
                if currency[cur].bs == 1: # 買い注文判定で売り注文決済
                    iden = orderIdSell[cur].pop(0)
                    closeOrder(oanda, ACCOUNT_ID, cur, iden)
                if currency[cur].bs == -1: # 売り注文判定で買い注文決済
                    iden = orderIdBuy[cur].pop(0)
                    closeOrder(oanda, ACCOUNT_ID, cur, iden)
                for i in range(len(orderHist[cur])): # 反対取引履歴の削除
                    if orderHist[cur][i] == -currency[cur].bs:
                        orderHist[cur][i] = 0
                        break
                currency[cur].bs = 0
            currency[cur].flagCrossOrder = False # 両建てフラグリセット

        # 注文
        for cur in CURRENCY:
            if not cur in tradeList: # 注文しない通貨ペア
                orderHist[cur].append(0)
                continue
            amount = int(currency[cur].orderAmount / len(tradeList))
            iden = order(oanda,ACCOUNT_ID, cur, currency[cur].bs, amount)
            if currency[cur].bs == 1:
                orderIdBuy[cur].append(iden)
            if currency[cur].bs == -1:
                orderIdSell[cur].append(iden)
            orderHist[cur].append(currency[cur].bs)
            currency[cur].bs = 0

        # 決済
        for cur in CURRENCY:
            bs = orderHist[cur].pop(0)
            if bs == 1:
                iden = orderIdBuy[cur].pop(0)
                closeOrder(oanda, ACCOUNT_ID, cur, iden)
            if bs == -1:
                iden = orderIdSell[cur].pop(0)
                closeOrder(oanda, ACCOUNT_ID, cur, iden)

        # 取引状況表示
        print(datetime.datetime.today())
        for cur in CURRENCY:
            print(cur)
            print(np.where(currency[cur].calTime()==1)[0][0])
            print(orderHist[cur])
            print(orderIdBuy[cur])
            print(orderIdSell[cur])
            print(tradeCurrency[cur])
        print()
