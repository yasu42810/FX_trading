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

# 取引通貨ペア
CURRENCY = ["USD_JPY", "EUR_JPY", "GBP_JPY", "EUR_USD",
            "NZD_JPY", "CHF_JPY", "CAD_JPY", "AUD_JPY"]

# 価格データ等
orderHist = {}
orderIdBuy = {}
orderIdSell = {}
tradeCurrency = {}
estLen = 20
for cur in CURRENCY:
    orderHist[cur] = [0]*(estLen-1)
    orderIdBuy[cur] = []
    orderIdSell[cur] = []
    tradeCurrency[cur] = False

# 取引パラメータ
threshold = {"USD_JPY":0.05, "EUR_JPY":0.05, "GBP_JPY":0.05, "EUR_USD":0.05,
             "NZD_JPY":0.05, "CHF_JPY":0.05, "CAD_JPY":0.05, "AUD_JPY":0.05}
# テスト
#threshold = {"USD_JPY":0.0001, "EUR_JPY":0.0001, "GBP_JPY":0.0001, "EUR_USD":0.0001,
#             "NZD_JPY":0.0001, "CHF_JPY":0.0001, "CAD_JPY":0.0001, "AUD_JPY":0.0001}
spreadReference = {"USD_JPY":0.023, "EUR_JPY":0.03, "GBP_JPY":0.035, "EUR_USD":0.025,
                   "NZD_JPY":0.03, "CHF_JPY":0.04, "CAD_JPY":0.025, "AUD_JPY":0.025}

# 通貨ペア事の取引時間
tradeTime = {"USD_JPY":[5, 16, 17, 18, 20], "EUR_JPY":[3, 4, 22, 23],
             "GBP_JPY":[22, 23], "EUR_USD":[0, 1, 4, 14, 15, 16, 18, 21, 22],
             "NZD_JPY":[10, 12, 13, 15, 18, 19, 23], "CHF_JPY":[23],
             "CAD_JPY":[5, 14, 16, 17, 18, 19, 20], "AUD_JPY":[17, 18, 19]}

# StandardScaler
usdjpy_ss = [joblib.load("model/usdjpy/StandardScaler/usdjpy_ss_"+str(i)+".sav") if i in tradeTime["USD_JPY"] else None for i in range(24)]
eurjpy_ss = [joblib.load("model/eurjpy/StandardScaler/eurjpy_ss_"+str(i)+".sav") if i in tradeTime["EUR_JPY"] else None for i in range(24)]
gbpjpy_ss = [joblib.load("model/gbpjpy/StandardScaler/gbpjpy_ss_"+str(i)+".sav") if i in tradeTime["GBP_JPY"] else None for i in range(24)]
eurusd_ss = [joblib.load("model/eurusd/StandardScaler/eurusd_ss_"+str(i)+".sav") if i in tradeTime["EUR_USD"] else None for i in range(24)]
nzdjpy_ss = [joblib.load("model/nzdjpy/StandardScaler/nzdjpy_ss_"+str(i)+".sav") if i in tradeTime["NZD_JPY"] else None for i in range(24)]
chfjpy_ss = [joblib.load("model/chfjpy/StandardScaler/chfjpy_ss_"+str(i)+".sav") if i in tradeTime["CHF_JPY"] else None for i in range(24)]
cadjpy_ss = [joblib.load("model/cadjpy/StandardScaler/cadjpy_ss_"+str(i)+".sav") if i in tradeTime["CAD_JPY"] else None for i in range(24)]
audjpy_ss = [joblib.load("model/audjpy/StandardScaler/audjpy_ss_"+str(i)+".sav") if i in tradeTime["AUD_JPY"] else None for i in range(24)]
ss = {"USD_JPY":usdjpy_ss, "EUR_JPY":eurjpy_ss, "GBP_JPY":gbpjpy_ss, "EUR_USD":eurusd_ss,
      "NZD_JPY":nzdjpy_ss, "CHF_JPY":chfjpy_ss, "CAD_JPY":cadjpy_ss, "AUD_JPY":audjpy_ss}

# NN
usdjpy_nn = [load_model("model/usdjpy/NN/usdjpy_nn_"+str(i)+".h5") if i in tradeTime["USD_JPY"] else None for i in range(24)]
eurjpy_nn = [load_model("model/eurjpy/NN/eurjpy_nn_"+str(i)+".h5") if i in tradeTime["EUR_JPY"] else None for i in range(24)]
gbpjpy_nn = [load_model("model/gbpjpy/NN/gbpjpy_nn_"+str(i)+".h5") if i in tradeTime["GBP_JPY"] else None for i in range(24)]
eurusd_nn = [load_model("model/eurusd/NN/eurusd_nn_"+str(i)+".h5") if i in tradeTime["EUR_USD"] else None for i in range(24)]
nzdjpy_nn = [load_model("model/nzdjpy/NN/nzdjpy_nn_"+str(i)+".h5") if i in tradeTime["NZD_JPY"] else None for i in range(24)]
chfjpy_nn = [load_model("model/chfjpy/NN/chfjpy_nn_"+str(i)+".h5") if i in tradeTime["CHF_JPY"] else None for i in range(24)]
cadjpy_nn = [load_model("model/cadjpy/NN/cadjpy_nn_"+str(i)+".h5") if i in tradeTime["CAD_JPY"] else None for i in range(24)]
audjpy_nn = [load_model("model/audjpy/NN/audjpy_nn_"+str(i)+".h5") if i in tradeTime["AUD_JPY"] else None for i in range(24)]
nn = {"USD_JPY":usdjpy_nn, "EUR_JPY":eurjpy_nn, "GBP_JPY":gbpjpy_nn, "EUR_USD":eurusd_nn,
      "NZD_JPY":nzdjpy_nn, "CHF_JPY":chfjpy_nn, "CAD_JPY":cadjpy_nn, "AUD_JPY":audjpy_nn}

# 時間関係
utc = pytz.utc
eastern = pytz.timezone("US/Eastern")
london = pytz.timezone("Europe/London")
nz = pytz.timezone("Pacific/Auckland")
zurich = pytz.timezone("Europe/Zurich")
sydney = pytz.timezone("Australia/Sydney")
tokyo = pytz.timezone("Asia/Tokyo")
timezone = {"USD_JPY":(utc, tokyo, eastern), "EUR_JPY":(utc, tokyo, london),
            "GBP_JPY":(utc, tokyo, london), "EUR_USD":(utc, tokyo, eastern),
            "NZD_JPY":(utc, tokyo, nz), "CHF_JPY":(utc, tokyo, zurich),
            "CAD_JPY":(utc, tokyo, eastern), "AUD_JPY":(utc, tokyo, sydney)}

nt = namedtuple("nt", "weekday hour minute")
stopOrderTime = nt(5, 4, 0)

class CurrencyManager():
    def __init__(self, api, accountId, currency, timezone, tradeTime,
                 thr, spreadReference, ss, nn):
        self.currency = currency
        self.utc, self.tzTokyo, self.tzRegion = timezone[self.currency]
        self.tradeTime = tradeTime[self.currency]
        self.ss = ss[self.currency]
        self.nn = nn[self.currency]
        self.api = api
        self.inputLen = 60
        self.estLen = 20
        self.para1 = 9
        self.para2 = 12
        self.para3 = 26
        self.thr = thr[self.currency]
        self.spread = 0
        self.spreadReference = spreadReference[self.currency]
        self.histPrice = self.setHistPrice
        self.histTechIndi = None
        self.inputData = None
        self.assetAmount = self.api.get_account(account_id=accountId)["balance"]
        self.needMargin = float(self.api.get_prices(instruments=self.currency)["prices"][0]["ask"] / 25)
        self.orderAmount = int((self.assetAmount / self.needMargin) / 30)
        self.orderAmount = 100
        self.flagCrossOrder = False

    # 過去の価格データの取得(初期化)
    def setHistPrice(self):
        count = self.inputLen + self.para1 + self.para3 - 2
        hist = self.api.get_history(instrument = self.currency, granularity = "M5", count = count)
        histPrice = []
        for i in range(count):
            histPrice.append(hist["candles"][i]["openAsk"])
        # ドルストレートの場合
        if self.currency == "EUR_USD":
            histPrice = [histPrice[i]*100 for i in range(len(histPrice))]
        self.histPrice = histPrice

    # スプレッドの取得
    def setSpread(self):
        while True:
            try:
                price = self.api.get_prices(instruments=self.currency)
                break
            except TypeError:
                print("TypeError")
            except Exception:
                print("Onother Error")
        # ドルストレートの場合
        if self.currency == "EUR_USD":
            self.spread = abs(price["prices"][0]["ask"] - price["prices"][0]["bid"]) * 100
        else:
            self.spread = abs(price["prices"][0]["ask"] - price["prices"][0]["bid"])

    # 最新の価格に更新
    def updataHistPrice(self):
        while True:
            try:
                price = self.api.get_history(instrument=self.currency, granularity="M5", count=1)
                break
            except TypeError:
                print("TypeError")
            except Exception:
                print("Onother Error")
        del self.histPrice[0]
        # ドルストレートの場合
        print(price)
        if self.currency == "EUR_USD":
            self.histPrice.append(price["candles"][0]["openAsk"] * 100)
        else:
            self.histPrice.append(price["candles"][0]["openAsk"])

    # RSI の計算
    def calRSI(self):
        df_diff = pd.DataFrame(self.histPrice).diff()
        def RSI(data):
            rise_sum = data[data>=0].sum()
            drop_sum = -data[data<0].sum()
            return rise_sum/(rise_sum+drop_sum)
        df_rsi = df_diff.rolling(window=14).apply(RSI)
        return df_rsi - 0.5

    # MACDの計算
    def calMACD(self):
        df_ema1 = pd.DataFrame(self.histPrice).ewm(span=self.para1).mean()
        df_ema2 = pd.DataFrame(self.histPrice).ewm(span=self.para2).mean()
        df_ema3 = pd.DataFrame(self.histPrice).ewm(span=self.para3).mean()
        df_macd = df_ema2 - df_ema3
        df_signal = df_macd.rolling(window=self.para1).mean()
        df_hist = df_macd - df_signal
        return pd.concat([df_macd, df_signal, df_hist], axis=1)

    # 乖離率の計算
    def calSepRate(self):
        df_open = pd.DataFrame(self.histPrice)
        df_ema2 = pd.DataFrame(self.histPrice).ewm(span=self.para2).mean()
        return (df_open - df_ema2) / df_ema2 * 100

    # テクニカルインディケータの計算
    def setTechIndi(self):
        df_open = pd.DataFrame(self.histPrice)
        df_macd = self.calMACD()
        df_rsi = self.calRSI()
        df_seprate = self.calSepRate()
        self.histTechIndi = pd.concat([df_open, df_macd, df_rsi, df_seprate], axis=1).iloc[-60:, :]

    # 現在の時間(東京時間から取引通貨国に変換)を計算
    def calTime(self):
        timeList = np.zeros(24)
        tokyoTime = self.tzTokyo.normalize(self.tzTokyo.localize(datetime.datetime.today()))
        utcTime = self.utc.normalize(tokyoTime.astimezone(self.utc))
        nowTime = self.tzRegion.normalize(utcTime.astimezone(self.tzRegion))
        timeList[nowTime.hour] = 1
        return timeList

    # NNに入力するデータの計算
    def setInputData(self):
        self.setTechIndi()
        inputData = copy.deepcopy(self.histTechIndi.iloc[:, 0].values).reshape(-1)
        inputData -= inputData.mean()
        inputData = np.append(inputData, self.histTechIndi.iloc[:, 1:].values.reshape(-1))
        inputData = np.delete(inputData, 0)
        self.setTime()
        print(self.currency, self.time)
        print(self.ss)
        inputData = self.ss[self.time].transform(inputData)
        """try:
            inputData = self.ss[self.time].transform(inputData)
        except:
            print(inputData)
            print(self.histPrice)
            print("ERROR!")
            exit()"""
        inputData = inputData.reshape(1, -1)
        self.inputData = inputData

    # 出力値の計算
    def setOutput(self):
        self.setTime()
        self.output = self.nn[self.time].predict(self.inputData)[0][0]
        print(self.currency+":"+str(self.output))

    # 取引時間であるかの判断(通貨毎の)
    def isTradeTime(self):
        #print(np.where(self.calTime()==1)[0][0])
        print(np.where(self.calTime()==1)[0][0], self.tradeTime)
        return np.where(self.calTime()==1)[0][0] in self.tradeTime

    # 現在時間のセット
    def setTime(self):
        self.time = np.where(self.calTime()==1)[0][0]

    # 出力値が閾値を超えいているか
    # 戻り値 : buy 1   stay 0    sell -1
    def overThr(self):
        if self.thr <= self.output:
            return 1
        elif -self.thr > self.output:
            return -1
        else:
            return 0

    # 売買判断
    # 戻り値 : 買いor売り True   ステイ False
    def canOrder(self):
        self.setSpread()
        if self.spread > self.spreadReference:
            return False
        self.bs = self.overThr()
        return self.bs != 0
