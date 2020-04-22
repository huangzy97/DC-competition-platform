# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:47:10 2019

@author: sbtithzy
"""
import pandas as pd
import numpy as np
import time,datetime
# 读取原始数据
def prepro():
    start = time.perf_counter()
    trian_data = pd.read_csv('./round1_diac2019_train.csv',low_memory=False)
    ###根据主订单id去重
    trian = trian_data.drop_duplicates(subset=['order_id'],keep='first')
    trian['count'] = 1
    customer_id = trian.groupby(['customer_id'],as_index=False)['count'].agg({'count':'count'})
    all_customer = pd.DataFrame(trian[['customer_id']]).drop_duplicates(['customer_id']).dropna()
    print(all_customer.shape)
    print('train date gap',trian.order_pay_time.max(),trian.order_pay_time.min())
    trian['order_pay_time'] = pd.to_datetime(trian['order_pay_time'])
    trian['order_pay_date'] = trian['order_pay_time'].dt.date

    # 预处理
    trian['is_customer_rate'] = trian['is_customer_rate'].fillna(0)
    trian['order_detail_amount'] = trian['order_detail_amount'].fillna(0)
    trian['customer_gender'] = trian['customer_gender'].fillna(method = 'ffill')
    trian['customer_province'] = trian['customer_province'].fillna(method = 'ffill')
    trian['customer_province'] = trian['customer_province'].fillna(method = 'bfill')
    trian['customer_city'] = trian['customer_city'].fillna(method = 'ffill')
    trian['customer_city'] = trian['customer_city'].fillna(method = 'bfill')
    trian['member_status'] = trian['member_status'].fillna(method = 'ffill')
    trian['is_member_actived'] = trian['is_member_actived'].fillna(method = 'ffill')
    ######编码转换
    trian['goods_delist_time'] = pd.to_datetime(trian['goods_delist_time'])
    trian['goods_list_time'] = pd.to_datetime(trian['goods_list_time'])
    trian['goods_time'] = trian['goods_delist_time'] - trian['goods_list_time']
    trian['goods_time'] = trian['goods_time']/np.timedelta64(1, 'D')
    last_time_data_1 = trian.groupby(['customer_id'],as_index=False)['order_pay_time'].agg({'order_pay_time':'min'})
    last_time_data_1['label'] = 'last'
    last_data_1 = pd.merge(trian,last_time_data_1,on = ['customer_id','order_pay_time'],how = 'left',copy=False)
    last_time_data_1 = last_data_1[last_data_1['label'] == 'last']###最后一次下单数据
    last_time_data_1 = last_time_data_1.drop_duplicates(subset=['customer_id'],keep='first')
    del last_time_data_1['label']
    left_data_1 = last_data_1.drop(last_data_1[last_data_1['label'] == 'last'].index) 
    del left_data_1['label']###剩余数据
    #############################取所有用户最后一次付款时间数据
    for i in range(2,max(customer_id['count'])+1):    
        locals()['last_time_data_'+str(i)] = locals()['left_data_'+str(i-1)].groupby(['customer_id'],as_index=False)['order_pay_time'].agg({'order_pay_time':'min'})
        locals()['last_time_data_'+str(i)]['label'] = 'last'
        locals()['last_data_'+str(i)] = pd.merge(locals()['left_data_'+str(i-1)],locals()['last_time_data_'+str(i)],on = ['customer_id','order_pay_time'],how = 'left',copy=False)
        locals()['last_time_data_'+str(i)] = locals()['last_data_'+str(i)][locals()['last_data_'+str(i)]['label'] == 'last']###最后一次下单数据
        locals()['last_time_data_'+str(i)] = locals()['last_time_data_'+str(i)].drop_duplicates(subset=['customer_id'],keep='first')
        del locals()['last_time_data_'+str(i)]['label']
        locals()['left_data_'+str(i)] = locals()['last_data_'+str(i)].drop(locals()['last_data_'+str(i)][locals()['last_data_'+str(i)]['label'] == 'last'].index) 
        del locals()['left_data_'+str(i)]['label']
    result = pd.merge(last_time_data_1,locals()['last_time_data_'+str(2)][['customer_id','order_pay_date']],on = 'customer_id',how = 'left',copy= False)
    result.rename(columns={'order_pay_date_x':'order_pay_date'},inplace=True)
    result.rename(columns={'order_pay_date_y':'order_pay_date_'+str(2)},inplace=True)
    for i in  range(3,max(customer_id['count'])+1): 
        result = pd.merge(result,locals()['last_time_data_'+str(i)][['customer_id','order_pay_date']],on = 'customer_id',how = 'left',copy= False)
        result.rename(columns={'order_pay_date_x':'order_pay_date'},inplace=True)
        result.rename(columns={'order_pay_date_y':'order_pay_date_'+str(i)},inplace=True)
    a = []
    result['delta_1'] = result['order_pay_date_2'] - result['order_pay_date']
    result['delta_1'] = result['delta_1'].dt.days + 1
    result['delta_1'] = result['delta_1'].fillna(0)
    a.append('delta_1')
    for i in range(2,max(customer_id['count'])):
        result['delta_'+str(i)] = result['order_pay_date_'+str(i+1)] - result['order_pay_date_'+str(i)]
        result['delta_'+str(i)] = result['delta_'+str(i)].dt.days + 1
        result['delta_'+str(i)] = result['delta_'+str(i)].fillna(0)
        a.append('delta_'+str(i))
    result['sum'] = result[a[0:len(a)]].apply(lambda x: x.sum(), axis=1)
    result = pd.merge(result,customer_id,on = 'customer_id',how = 'left',copy = False)
    result.rename(columns= {'count_y':'count'},inplace = True)
    result.rename(columns= {'order_pay_date':'order_pay_date_1'},inplace = True)
    result['average'] = result['sum']/result['count']
    result['min_time_point'] = result['order_pay_date_1'] + datetime.timedelta(days=180)
    result['max_time_point'] = pd.to_datetime('2014-01-01')+datetime.timedelta(days=180)
    trian = pd.merge(trian,result[['customer_id','order_pay_date_1','order_pay_date_2','order_pay_date_3','min_time_point','max_time_point','count','sum','average']],on = 'customer_id',how = 'left',copy = False)
    trian.rename(columns= {'count_y':'count_number'},inplace = True)
    elapsed = (time.perf_counter() - start)#####时间结束点
    print (u'预处理累计用时:',elapsed) #####累计用时
    return trian_data,trian,all_customer