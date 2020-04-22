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
batch_size = 128
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
drop = ['sale_date','sale_date_1']#需要删除的列
data.drop(drop,axis=1,inplace=True)
# 哑变量

le = LabelEncoder()
var_to_encode = ['sale_date_2','year_built','year_repair']
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
#  数据拆分
train_data_re = data[data['label']=='train']
test_data_re = data[data['label']=='test']
#
train_data_re.drop(['label','ID'],axis = 1,inplace=True)
test_data_re.drop(['label','ID','price'],axis = 1,inplace=True)
# 模型选择
feature = []
for i in train_data_re.columns:
    if i == 'price':
        pass
    else:
        feature.append(i) 
X = train_data_re[feature]
y = train_data_re[target]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 42)
# =============================================================================
# from sklearn.linear_model import LinearRegression
# LR = LinearRegression()
# LR.fit(X_train,y_train)
# predict = LR.predict(test_data_re)
# print(LR.score(X_train,y_train))###模型训练集分数
# print(LR.score(X_test,y_test))#验证集集分数
# =============================================================================
# 预测

# # 划分训练集和验证集
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.05, random_state=42,stratify=y)
# 
# =============================================================================
# submit_df = submit[['customer_id']]
# X_submit = submit[feature]
# =============================================================================
#y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test, nb_classes)

# Dense层:即全连接层
# keras.layers.core.Dense(output_dim, init='glorot_uniform', activation='linear', weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
###6个隐藏层,初始化权值为随机 ,激活函数使用线性整流函数 ,输入层的个数为11
model = Sequential()
model.add(Dense(units = 128,kernel_initializer='uniform',activation='relu',input_dim = 205))
###再添加一个隐藏
model.add(Dense(units = 64,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units = 32,kernel_initializer='uniform',activation='relu'))
model.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
###添加一个输出层
model.add(Dense(units = 1,kernel_initializer='uniform'))
# =============================================================================
# 编译ANN
# 编译ANN的时候重要参数,optimizer,loss,这里选择随机梯度下降,损失函数选择 binary_crossentropy,
# 主要考虑输出是二分类的数据， metrics这里关注准确率 
# =============================================================================
#model.compile(optimizer = 'adam',loss = 'mse',metrics = ['accuracy'])
# Dropout  需要断开的连接的比例
#model.add(Dropout(0.2))

# 打印出模型概况
print('model.summary:')
model.summary()

# 在训练模型之前，通过compile来对学习过程进行配置
# 编译模型以供训练
# 包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']
# 如果要在多输出模型中为不同的输出指定不同的指标，可像该参数传递一个字典，例如metrics={'ouput_a': 'accuracy'}
model.compile(loss='mse',optimizer='adam',metrics=['mse'])
#classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])


# 训练模型
# Keras以Numpy数组作为输入数据和标签的数据类型
# fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[], validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None)
# nb_epoch：整数，训练的轮数，训练数据将会被遍历nb_epoch次。Keras中nb开头的变量均为"number of"的意思
# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# shuffle：布尔值，表示是否在训练过程中每个epoch前随机打乱输入样本的顺序。

# fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况
history = model.fit(X_train, y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, y_test))


# evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
# 按batch计算在某些输入数据上模型的误差

print('-------evaluate--------')
score = model.evaluate(X_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
predict = model.predict(test_data_re)
##########################################################
test_data_re['price'] = predict
test_data_re['ID'] = ID
# 输出结果
test_data_re[['ID','price']].to_csv(r'D:/dm/DC竞赛/美国King County房价预测训练赛/predict.csv', index=False)
