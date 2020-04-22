# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:05:37 2019

@author: sbtithzy
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import datetime
def make_feature_and_label(date1,date2,isSubmit):
    date1['count'] = 1
    data = date1[['customer_id','order_pay_date_1','order_pay_date_2','order_pay_date_3','count','sum','average']].drop_duplicates(subset=['customer_id'],keep='first')
    # 统计这个用户出现了多少次
    customer_id = date1.groupby(['customer_id'],as_index=False)['count'].agg({'count':'count'})
# 统计这个用户购买商品的价格信息
    good_price = date1.groupby(['customer_id'],as_index=False)['goods_price'].agg({'goods_price_max':'max','goods_price_min':'min','goods_price_mean':'mean'})
# 统计这个用户的订单最后一次购买时间
    last_time = date1.groupby(['customer_id'],as_index=False)['order_pay_date'].agg({'order_pay_date_last':'max','order_pay_date_first':'min'})
# customer_province
# =============================================================================
    order_total_discount = date1.groupby(['customer_id'],as_index=False)['order_total_discount'].agg({'order_total_discount_max':'max','order_total_discount_sum':'sum','order_total_discount_mean':'mean'})
# 是否支持折扣 goods_has_discount
    goods_has_discount_0 = date1[date1['goods_has_discount']==0].groupby(['customer_id'],as_index=False)['count'].agg({'count_goods_has_discount_0':'count'})
    goods_has_discount_1 = date1[date1['goods_has_discount']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_goods_has_discount_1':'count'})
    goods_has_discount = pd.merge(goods_has_discount_0,goods_has_discount_1,on=['customer_id'],how='outer',copy=False)   
    goods_has_discount = goods_has_discount.fillna(0)
# 商品状态    
    goods_status_1 = date1[date1['goods_status']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_goods_status_1':'count'})
    goods_status_2 = date1[date1['goods_status']==2].groupby(['customer_id'],as_index=False)['count'].agg({'count_goods_status_2':'count'})
    goods_status = pd.merge(goods_status_1,goods_status_2,on=['customer_id'],how='outer',copy=False)   
    goods_status = goods_status.fillna(0)
# 会员状态
# =============================================================================
#     member_status_1 = date1[date1['member_status']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_member_status_1':'count'})
#     member_status_2 = date1[date1['member_status']==2].groupby(['customer_id'],as_index=False)['count'].agg({'count_member_status_2':'count'})
#     member_status_3 = date1[date1['member_status']==3].groupby(['customer_id'],as_index=False)['count'].agg({'count_member_status_3':'count'})
#     member_status = pd.merge(member_status_1,member_status_2,on=['customer_id'],how='outer',copy=False)
#     member_status = pd.merge(member_status,member_status_3,on=['customer_id'],how='outer',copy=False)                                                                                  
#     member_status = member_status.fillna(0)
#     for i in range(member_status.shape[0]):
#         idx = np.argmax(member_status.iloc[i], axis=1)
#         member_status[idx][i] = 1
#         member_status[member_status != 1]=0     
# =============================================================================
# =============================================================================
#     member_status_1 = date1.groupby(['customer_id','member_status'],as_index=False)['count'].agg({'count':'count'})
#     member_status_2 = member_status_1[['customer_id','count']].groupby(by='customer_id',as_index=False).max()
#     member_status = pd.merge(member_status_2,member_status_1,on=['customer_id','count'],how='left')
#     del member_status['count']
#     member_status = member_status.drop_duplicates(subset = ['customer_id'], keep = 'first')
#     
# =============================================================================
# 会员是否激活                                                                                    
# =============================================================================
#     is_member_actived_0 = date1[date1['is_member_actived']==0].groupby(['customer_id'],as_index=False)['count'].agg({'count_is_member_actived_0':'count'})
#     is_member_actived_1 = date1[date1['is_member_actived']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_is_member_actived_1':'count'})
#     is_member_actived = pd.merge(is_member_actived_0,is_member_actived_1,on=['customer_id'],how='outer',copy=False)
#     is_member_actived = is_member_actived.fillna(0)
#     for i in range(is_member_actived.shape[0]):
#         idx = np.argmax(is_member_actived.iloc[i], axis=1)
#         is_member_actived[idx][i] = 1
#         is_member_actived[is_member_actived != 1]=0    
# =============================================================================
# =============================================================================
#     is_member_actived_0 = date1.groupby(['customer_id','is_member_actived'],as_index=False)['count'].agg({'count':'count'})
#     is_member_actived_1 = is_member_actived_0[['customer_id','count']].groupby(by='customer_id',as_index=False).max()
#     is_member_actived = pd.merge(is_member_actived_1,is_member_actived_0,on=['customer_id','count'],how='left')
#     del is_member_actived['count']
#     is_member_actived = is_member_actived.drop_duplicates(subset = ['customer_id'], keep = 'first')
#     
# =============================================================================
# 性别
# =============================================================================
#     customer_gender_0 = date1[date1['customer_gender']==0].groupby(['customer_id'],as_index=False)['count'].agg({'count_customer_gender_0':'count'})
#     customer_gender_1 = date1[date1['customer_gender']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_customer_gender_1':'count'})
#     customer_gender_2 = date1[date1['customer_gender']==2].groupby(['customer_id'],as_index=False)['count'].agg({'count_customer_gender_2':'count'})
#     customer_gender = pd.merge(customer_gender_0,customer_gender_1,on=['customer_id'],how='outer',copy=False)
#     customer_gender = pd.merge(customer_gender,customer_gender_2,on=['customer_id'],how='outer',copy=False)
#     customer_gender = customer_gender.fillna(0)
#     for i in range(customer_gender.shape[0]):
#         idx = np.argmax(customer_gender.iloc[i], axis=1)
#         customer_gender[idx][i] = 1
#         customer_gender[customer_gender != 1]=0
# =============================================================================
    customer_gender_0 = date1.groupby(['customer_id','customer_gender'],as_index=False)['count'].agg({'count':'count'})
    customer_gender_1 = customer_gender_0[['customer_id','count']].groupby(by='customer_id',as_index=False).max()
    customer_gender = pd.merge(customer_gender_1,customer_gender_0,on=['customer_id','count'],how='left')
    del customer_gender['count']
    customer_gender = customer_gender.drop_duplicates(subset = ['customer_id'], keep = 'first')
# 是否评价(评价次数)
    is_customer_rate_0 = date1[date1['is_customer_rate']==0].groupby(['customer_id'],as_index=False)['count'].agg({'count_is_customer_rate_0':'count'})
    is_customer_rate_1 = date1[date1['is_customer_rate']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_is_customer_rate_1':'count'})
    is_customer_rate = pd.merge(is_customer_rate_0,is_customer_rate_1,on=['customer_id'],how='outer',copy=False)
    is_customer_rate = is_customer_rate.fillna(0)

# 子订单状态
# =============================================================================
#     order_detail_status_1 = date1[date1['order_detail_status']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_1':'count'})
#     order_detail_status_2 = date1[date1['order_detail_status']==2].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_2':'count'})
#     order_detail_status_3 = date1[date1['order_detail_status']==3].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_3':'count'})
#     order_detail_status_4 = date1[date1['order_detail_status']==4].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_4':'count'})
#     order_detail_status_5 = date1[date1['order_detail_status']==5].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_5':'count'})
#     order_detail_status_6 = date1[date1['order_detail_status']==6].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_detail_status_6':'count'})
#     order_detail_status = pd.merge(order_detail_status_1,order_detail_status_2,on=['customer_id'],how='outer',copy=False)
#     order_detail_status = pd.merge(order_detail_status,order_detail_status_3,on=['customer_id'],how='outer',copy=False)
#     order_detail_status = pd.merge(order_detail_status,order_detail_status_4,on=['customer_id'],how='outer',copy=False)
#     order_detail_status = pd.merge(order_detail_status,order_detail_status_5,on=['customer_id'],how='outer',copy=False)
#     order_detail_status = pd.merge(order_detail_status,order_detail_status_6,on=['customer_id'],how='outer',copy=False)    
#     order_detail_status = order_detail_status.fillna(0)
# 
# # 订单状态
#     order_status_1 = date1[date1['order_status']==1].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_1':'count'})
#     order_status_2 = date1[date1['order_status']==2].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_2':'count'})
#     order_status_3 = date1[date1['order_status']==3].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_3':'count'})
#     order_status_4 = date1[date1['order_status']==4].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_4':'count'})
#     order_status_5 = date1[date1['order_status']==5].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_5':'count'})
#     order_status_6 = date1[date1['order_status']==6].groupby(['customer_id'],as_index=False)['count'].agg({'count_order_status_6':'count'})
#     order_status = pd.merge(order_status_1,order_status_2,on=['customer_id'],how='outer',copy=False)
#     order_status = pd.merge(order_status,order_status_3,on=['customer_id'],how='outer',copy=False)
#     order_status = pd.merge(order_status,order_status_4,on=['customer_id'],how='outer',copy=False)
#     order_status = pd.merge(order_status,order_status_5,on=['customer_id'],how='outer',copy=False)
#     order_status = pd.merge(order_status,order_status_6,on=['customer_id'],how='outer',copy=False)
#     order_status = order_status.fillna(0)
# # 商品上架时长
# =============================================================================
    goods_time = date1.groupby(['customer_id'],as_index=False)['goods_time'].agg({ 'goods_time_sum':'sum'})
#     
    order_total_num = date1.groupby(['customer_id'],as_index=False)['order_total_num'].agg({'order_total_num_sum':'sum'})
    data = pd.merge(data,customer_id,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,good_price,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,last_time,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,order_total_discount,on=['customer_id'],how='left',copy=False)
#    data = pd.merge(data,customer_province,on=['customer_id'],how='left',copy=False)
#    data = pd.merge(data,is_member_actived,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,customer_gender,on=['customer_id'],how='left',copy=False)
#    data = pd.merge(data,member_status,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,is_customer_rate,on=['customer_id'],how='left',copy=False)
#    data = pd.merge(data,order_status,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,goods_has_discount,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,goods_time,on=['customer_id'],how='left',copy=False)
    data = pd.merge(data,order_total_num,on=['customer_id'],how='left',copy=False)  
    data = pd.merge(data,goods_status,on=['customer_id'],how='left',copy=False) 
    le = LabelEncoder()
    var_to_encode = ['customer_gender']
    for col in var_to_encode:
        data[col] = le.fit_transform(data[col].astype(str))
    data = pd.get_dummies(data, columns=var_to_encode)
    del data['order_pay_date_1']
    del data['order_pay_date_2']
    del data['order_pay_date_3']
    del data['order_pay_date_first']
    if isSubmit==False:
        data['order_pay_date_last'] = pd.to_datetime(date2['order_pay_date'].min()) - pd.to_datetime(data['order_pay_date_last'])
        data['order_pay_date_last'] = data['order_pay_date_last'].dt.days + 1
        data['label'] = 0
        data.loc[data['customer_id'].isin(list(date2['customer_id'].unique())),'label'] = 1
        print(data['label'].mean())
    else:
        data['order_pay_date_last'] = pd.to_datetime('2013-12-31') - pd.to_datetime(data['order_pay_date_last'])
        data['order_pay_date_last'] = data['order_pay_date_last'].dt.days + 1
    print(data.shape)
    return data