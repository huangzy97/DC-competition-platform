# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:15:22 2019

@author: sbtithzy
"""

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from keras.utils import np_utils
import warnings
warnings.filterwarnings('ignore')

def DNN(X_train,y_train,X_valid,y_valid,X_submit,batch_size,nb_epoch,input_num):
    model = Sequential()
    model.add(Dense(units = 128,kernel_initializer='uniform',activation='relu',input_dim = input_num))
    ###再添加一个隐藏
    model.add(Dense(units = 32,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 8,kernel_initializer='uniform',activation='relu'))
    ###添加一个输出层
    #model.add(Dense(units = 4,kernel_initializer='uniform',activation='relu'))
    model.add(Dense(units = 1,kernel_initializer='uniform',activation='sigmoid'))
    # =============================================================================
    # 主要考虑输出是二分类的数据， metrics这里关注准确率 
    # =============================================================================
    model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics = ['accuracy'])
    # Dropout  需要断开的连接的比例
    #model.add(Dropout(0.2))
    
    # 打印出模型概况
    print('model.summary:')
    model.summary()  
    history = model.fit(X_train, y_train,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1, validation_data=(X_valid, y_valid))
    y_submit = model.predict(X_submit)
    return y_submit