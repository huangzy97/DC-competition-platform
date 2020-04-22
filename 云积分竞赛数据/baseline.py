# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 21:09:49 2019

@author: sbtithzy
"""
'''
比赛数据（脱敏后）抽取的是一段时间范围内，客户的购买行为数据，初赛和复赛两个阶段所提供样本的量级有所不同。
参赛选手根据特征字段信息进行建模，预测所有已购客户在未来180天的购买概率。
'''
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 10:15:02 2019

@author: sbtithzy
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn import model_selection 
from sklearn.model_selection import train_test_split
import datetime
import warnings
import time 
import make_feature_and_label,preprocess,lgb,DNN
warnings.filterwarnings('ignore')

batch_size = 100
nb_classes = 10
nb_epoch = 20
# 读取预处理后数据
print(u'预处理部分')
trian_data,trian,all_customer = preprocess.prepro()
#result['min_time_point'] = result['order_pay_date_1'] + datetime.timedelta(days=180)
#result['max_time_point'] = pd.to_datetime('2014-01-01')+datetime.timedelta(days=180)
validata_date_begin = trian['order_pay_date'].max() - datetime.timedelta(days=180)
# 简单的特征生成部分代码
# 生成训练数据和提交数据
train_history = trian[(trian['order_pay_date'].astype(str)<='2013-06-30')]
online_history = trian[(trian['order_pay_date'].astype(str)<='2013-12-31')]
# train_label 相对于 train_history 的未来180天的数据
train_label = trian[trian['order_pay_date'].astype(str)>='2013-07-01']
#
print(u'特征工程及训练集/测试集部分')
start = time.perf_counter()
train = make_feature_and_label.make_feature_and_label(train_history,train_label,False)
submit = make_feature_and_label.make_feature_and_label(online_history,None,True)
elapsed = (time.perf_counter() - start)#####时间结束点
print (u'特征工程及训练集/测试集累计用时:',elapsed) #####累计用时
# 模型准备

submit_df = submit[['customer_id']]
test = submit

target= 'label'
IDcol = 'customer_id'
######测试模型建立
def modelfit(alg, dtrain, dtest, predictors1,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    global all_customer,submit_df
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors1].values, label=dtrain[target].values)
        xgtest = xgb.DMatrix(dtest[predictors1].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)
        print (cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    #Fit the algorithm on the data
    alg.fit(dtrain[predictors1], dtrain['label'],eval_metric='auc')#error

    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors1])
    
    print(dtrain_predictions)
    dtrain_predprob = alg.predict_proba(dtrain[predictors1])[:,1]
    #Print model report:
    print ("\nModel Report")
    print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))
#####预测
    dtest_predictions = alg.predict_proba(dtest[predictors])[:,1]
    submit_df['result'] = dtest_predictions 
    all_customer = pd.merge(all_customer,submit_df,on=['customer_id'],how='left',copy=False)
    all_customer = all_customer.sort_values(['customer_id'])
    all_customer['customer_id'] = all_customer['customer_id'].astype('int64')
    all_customer['result'] = all_customer['result'].fillna(0)
    all_customer.to_csv('./hzy_baseline.csv',index=False)
    xgb_fea_imp=pd.DataFrame(list(alg.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
    print('',xgb_fea_imp)
predictors = [x for x in train.columns if x not in [target, IDcol]]
####最终模型的参数
xgb1 = XGBClassifier(
        learning_rate =0.1,
        n_estimators=2000,
        max_depth=5,
        min_child_weight=8,
        gamma=0.3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        scale_pos_weight=1,
        seed=27)
print(modelfit(xgb1, train, test, predictors))
