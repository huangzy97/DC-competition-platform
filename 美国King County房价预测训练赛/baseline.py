# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:39:52 2020

@author: huangzy97
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# load test and train data
train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test_nolabel.csv')
#
batch_size = 100
nb_classes = 10
nb_epoch = 100
ID = test_data['ID']
target = ['price']
####
train_data['label'] = 'train'
test_data['label'] = 'test'
data = pd.concat([train_data,test_data],axis = 0)
# data process
data['sale_date_1'] = data['sale_date'].astype(str).str[0:4].astype(int)###取房屋出售年份
data['house_age'] = data['sale_date_1'] - data['year_built'] ##生成房屋的年龄
data['is_repair'] = data['year_repair'].apply(lambda x: 0 if x==0 else 1)       #生成是否维修特征   
data['sale_date_2'] = data['sale_date'].astype(str).str[0:6].astype(int)  #哪个月出售
data['area_parking'] = data['area_parking'].apply(lambda x: data['area_parking'].mean() if x >500000 else x)# 异常点处理
drop = ['year_repair','sale_date','sale_date_1','year_built']#需要删除的列
data.drop(drop,axis=1,inplace=True)
# 哑变量
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
var_to_encode = ['sale_date_2']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
data = pd.get_dummies(data, columns=var_to_encode)
columns = ['area_basement','area_house','area_parking','floorage','latitude','longitude','num_bathroom','num_bedroom','rating','house_age',
           'floor']
####归一化
# =============================================================================
# for i in columns:
#     data[i] = (data[i]-data[i].min())/(data[i].max() - data[i].min())
# =============================================================================
for i in columns:
    data[i] = (data[i]-data[i].min())/data[i].std()
#
x = data['is_repair']
y = data['price']
plt.scatter(x,y)
plt.show
#
#num_bedroom，num_bathroom，floor，rating，floorage
x = data['longitude']
y = data['price']
plt.scatter(x,y)
plt.show
#
#  数据拆分
train_data_re = data[data['label']=='train']
test_data_re = data[data['label']=='test']
#
train_data_re.drop(['label','ID'],axis = 1,inplace=True)
test_data_re.drop(['label','ID','price'],axis = 1,inplace=True)
# 模型选择
feature = ['area_basement', 'area_house', 'area_parking', 'floor',
       'floorage', 'num_bathroom', 'num_bedroom','latitude','longitude',
        'rating', 'house_age', 'is_repair', 'sale_date_2_0',
       'sale_date_2_1', 'sale_date_2_2', 'sale_date_2_3', 'sale_date_2_4',
       'sale_date_2_5', 'sale_date_2_6', 'sale_date_2_7', 'sale_date_2_8',
       'sale_date_2_9', 'sale_date_2_10', 'sale_date_2_11', 'sale_date_2_12']
X = train_data_re[feature]
y = train_data_re[target]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(X_train,y_train)
predict = LR.predict(test_data_re)
print(LR.score(X_train,y_train))###模型训练集分数
print(LR.score(X_test,y_test))#验证集集分数
# 预测
predict = LR.predict(test_data_re)
test_data_re['price'] = predict
test_data_re['ID'] = ID
# 输出结果
test_data_re[['ID','price']].to_csv(r'D:/dm/DC竞赛/美国King County房价预测训练赛/predict.csv', index=False)