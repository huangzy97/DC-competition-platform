# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 09:39:52 2020

@author: huangzy97
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
import warnings
warnings.filterwarnings('ignore')
# load test and train data
train_data = pd.read_csv('./train.csv')
test_data  = pd.read_csv('./test_nolabel.csv')
#
batch_size = 32
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

le = LabelEncoder()
var_to_encode = ['sale_date_2']
for col in var_to_encode:
    data[col] = le.fit_transform(data[col])
data = pd.get_dummies(data, columns=var_to_encode)
columns = ['area_basement','area_house','area_parking','floorage',
           'latitude','longitude','num_bathroom','num_bedroom','rating',
           'house_age','floor']
####归一化
# =============================================================================
# for i in columns:
#     data[i] = (data[i]-data[i].min())/(data[i].max() - data[i].min())
# =============================================================================
for i in columns:
    data[i] = (data[i]-data[i].mean())/data[i].std()
#
# =============================================================================
# x = data['is_repair']
# y = data['price']
# plt.scatter(x,y)
# plt.show
# #
# #num_bedroom，num_bathroom，floor，rating，floorage
# x = data['longitude']
# y = data['price']
# plt.scatter(x,y)
# plt.show
# =============================================================================
#
# 数据拆分
train_data_re = data[data['label']=='train']
test_data_re = data[data['label']=='test']
#
train_data_re.drop(['label','ID'],axis = 1,inplace=True)
test_data_re.drop(['label','ID','price'],axis = 1,inplace=True)
# 整理训练和验证数据集
feature = []
for i in train_data_re.columns:
    if i == 'price':
        pass
    else:
        feature.append(i) 
X = train_data_re[feature]
y = train_data_re[target]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)
### 添加网络模型
model = Sequential()
model.add(Dense(units = 128,kernel_initializer='uniform',activation='relu',input_dim = len(feature)))
###再添加一个隐藏
#model.add(Dropout(0.5))
model.add(Dense(units = 64,kernel_initializer='uniform',activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(units = 32,kernel_initializer='uniform',activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(units = 16,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
###添加一个输出层
model.add(Dense(units = 1,kernel_initializer='uniform'))
#model.add(Dropout(0.2))
# 打印出模型概况
print('model.summary:')
model.summary()
# 在训练模型之前，通过compile来对学习过程进行配置，编译模型以供训练
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
# 训练模型
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))
# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# 按batch计算在某些输入数据上模型的误差
print('-------train--------')
train_score = model.evaluate(X_train, y_train, verbose=0)
print('Train loss:', train_score[0])
predict = model.predict(test_data_re)
print('-------evaluate--------')
test_score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', test_score[0])
# 预测结果
predict = model.predict(test_data_re)
test_data_re['price'] = predict
test_data_re['ID'] = ID
# 输出结果
test_data_re[['ID','price']].to_csv(r'D:/dm/DC竞赛/美国King County房价预测训练赛/predict.csv', index=False)
