# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:12:18 2019

@author: sbtithzy
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import datetime
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')
# 参数
param = {
    'num_leaves':128,
    'objective':'binary',
    'max_depth':-1,
    'learning_rate':0.002,
    'metric':'binary_logloss'}
# 构建机器学习所需的label和data
def LGB(X_train,y_train,X_valid,y_valid,X_submit,num_boost_round,early_stopping_rounds):
    trn_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_valid, label=y_valid)
    lgbm = lgb.train(param,trn_data,valid_sets=[trn_data,val_data],num_boost_round = num_boost_round ,early_stopping_rounds=early_stopping_rounds,verbose_eval=50)
    y_submit = lgbm.predict(X_submit)
    y_pred = lgbm.predict(X_valid)
    y_pred = (y_pred > 0.45)
    cm = confusion_matrix(y_valid, y_pred)
    recall = cm[0][0]/(cm[0][0]+cm[0][1])
    prec = cm[0][0]/(cm[0][0]+cm[1][1])
    F1_score = 2*recall*prec/(recall+prec)
    print ('精准率：',recall)
    print ('召回率：',prec)
    print ('F1_SCORE:',F1_score)
    return y_submit
