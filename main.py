import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from model import MyLGB
from agent import Agent

root='./'
bad_stocks=[]
agent=Agent()

## 下载沪深300指数
# agent.download_300(save_dir='./data/', start_date='2002-01-01', 
#                    end_date='2021-12-31', frequency="d")

## 下载沪深300成分股
# agent.download_data(save_dir='./data/raw_data/', start_date='2002-01-01', 
#                     end_date='2021-12-31', frequency="d")


##制作FE后的沪深300数据集作为后面模型的input_data
# hs300_stocks=os.listdir('./data/raw_data/')
# for stock in tqdm(hs300_stocks):
#     data=pd.read_csv(os.path.join(root,'data/raw_data',stock),index_col=0,parse_dates=True)
#     data.rename(columns={'pctChg': 'return'}, inplace=True) 
#     try:
#         FE_data=agent.feature_engineering(data)
#         FE_data=FE_data.dropna()
#         # FE_data=FE_data.interpolate(method='linear', limit_direction='both',axis=0)
#         FE_data.to_csv(os.path.join(root,'data/input_data',stock))
#     except:
#         print('bad_stock:',stock)
#         bad_stocks.append(stock)



##读取上面的input_data，划分数据集并训练模型
# hs300_stocks=os.listdir('./data/input_data/')
# val_res=[]
# for stock in tqdm(hs300_stocks):
#     FE_data=pd.read_csv(os.path.join(root,'data/input_data',stock),index_col=0,parse_dates=True)
#     try:
#         x_train, x_val, x_test, y_train, y_val, y_test=agent.make_dataset(FE_data)   
#         # create_model===========================================================        
#         mylgb=MyLGB()
#         best_model=mylgb.train(x_train, y_train, x_val, y_val)
#         mylgb.save(best_model,model_path=os.path.join(root,'model',stock.replace('.csv','.pkl')))



##加载训练好的模型，在验证集上筛选出预测较准确的票
# hs300_stocks=os.listdir('./data/input_data/')
# val_res=[]
# for stock in tqdm(hs300_stocks):
#     FE_data=pd.read_csv(os.path.join(root,'data/input_data',stock),index_col=0,parse_dates=True)
#     try:
#         x_train, x_val, x_test, y_train, y_val, y_test=agent.make_dataset(FE_data)   
#         # load_model===========================================================        
#         mylgb=MyLGB()
#         best_model=mylgb.load(model_path=os.path.join(root,'model',stock.replace('.csv','.pkl')))               
#         y_pred=mylgb.predict(best_model, x_val)
#         # eval_res===========================================================
#         res=agent.get_res(y_pred, y_val)
#         error=agent.eval_res(res)  ##平均每天预测误差，0.004表示平均每天误差0.4%
#         val_res.append([stock,error])
#     except:
#         print('bad_stock:',stock)
#         bad_stocks.append(stock)    
# val_res.sort(key=lambda x:x[1]) ##按在val上的预测error从低到高排序
# my_stocks=pd.DataFrame(val_res[:100],columns=['stock','error']) ##取top100的票加入池子
# my_stocks.to_csv('data/my_stocks.csv', index=False)



##加载训练好的模型，在测试集上查看预测误差，并且寻找公共数据时间段
# hs300_stocks=os.listdir('./data/input_data/')
# my_stocks=pd.read_csv('data/my_stocks.csv')
# test_res=[]
# date=pd.DataFrame(columns=['date'])
# for stock in tqdm(hs300_stocks):
#     if stock not in list(my_stocks['stock']): ##只考虑池子里的票
#         continue
#     FE_data=pd.read_csv(os.path.join(root,'data/input_data',stock),index_col=0,parse_dates=True)
#     try:
#         x_train, x_val, x_test, y_train, y_val, y_test=agent.make_dataset(FE_data)   
#         if len(y_test)<200:  ##跳过数据太少的股票
#             bad_stocks.append(stock) 
#             continue
#         y_test.name=stock
#         date = pd.merge(date, y_test, on='date', how='outer')
#         date=date.dropna()
#         # load_model===========================================================        
#         mylgb=MyLGB()
#         best_model=mylgb.load(model_path=os.path.join(root,'model',stock.replace('.csv','.pkl')))               
#         y_pred=mylgb.predict(best_model, x_test)
#         # eval_res===========================================================
#         res=agent.get_res(y_pred, y_test)
#         error=agent.eval_res(res)  ##平均每天预测误差，0.004表示平均每天误差0.4%
#         test_res.append([stock,error])
#     except:
#         print('bad_stock:',stock)
#         bad_stocks.append(stock)    
# test_res.sort(key=lambda x:x[1]) ##按在test上的预测error从低到高排序



##保存数据或其它方面有问题的stock名和公共数据时间段
# df = pd.DataFrame(bad_stocks, columns=['Stock'])
# df.to_csv('data/bad_stock.csv', index=False)
# date['date'].to_csv('data/available_test_date.csv', index=False)

##制作好回测用的三维（股票数*时间数*特征数）数据集和参照物沪深300指数
# hs300_stocks=os.listdir('./data/input_data/')
# my_stocks=pd.read_csv('data/my_stocks.csv')
# bad_stock=pd.read_csv('data/bad_stock.csv')
# index300=pd.read_csv('data/300.csv')
# available_test_date=pd.read_csv('data/available_test_date.csv')
# available_test_date.index=available_test_date['date']
# testdata=[]
# for stock in tqdm(hs300_stocks):
#     if stock not in list(my_stocks['stock']) or stock in list(bad_stock['Stock']): ##只考虑池子里的票
#         continue
#     FE_data=pd.read_csv(os.path.join(root,'data/input_data',stock),index_col=0,parse_dates=False)

#     x_train, x_val, x_test, y_train, y_val, y_test=agent.make_dataset(FE_data)   
#     x_test=x_test.join(available_test_date, on='date', how='inner') ##统一测试时间段
#     del x_test['date']
#     # print(x_test.shape)
#     # y_test=pd.DataFrame(y_test).join(available_test_date, on='date', how='inner') ##统一测试时间段
#     testdata.append([stock,x_test])
# with open('data/testdata.pkl', 'wb') as file: ##多维复杂list保存
#     pickle.dump(testdata, file)
    


##每agent.seq_len=30 ##月频调仓一次
file_path = 'data/testdata.pkl'# 加载以 .pkl 格式保存的文件
with open(file_path, 'rb') as file:
    testdata = pickle.load(file)
intervals=len(testdata[0][1])//agent.seq_len

for i in tqdm(range(intervals)-1):
    factors=[] ##模型预测结果可以看成综合因子
    for stock,x_test in testdata:
        x_test=x_test.iloc[i*agent.seq_len:(i+1)*agent.seq_len]
        mylgb=MyLGB()
        best_model=mylgb.load(model_path=os.path.join(root,'model',stock.replace('.csv','.pkl')))               
        y_pred=mylgb.predict(best_model, x_test)       
        factors.append([stock,(y_pred['pred']/100+1).prod()])
    factors.sort(key=lambda x:x[1],reverse=True) ##按同一时间段，池子里的股票综合因子从高到低排序
    
        

