import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import MyLGB
from agent import Agent

root='./'
bad_stocks=[]
agent=Agent()


##下载沪深300原始data
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
hs300_stocks=os.listdir('./data/input_data/')
for stock in tqdm(hs300_stocks):
    FE_data=pd.read_csv(os.path.join(root,'data/input_data',stock),index_col=0,parse_dates=True)
    try:
        x_train, x_val, x_test, y_train, y_val, y_test=agent.make_dataset(FE_data)   
        # create_model===========================================================        
        mylgb=MyLGB()
        best_model=mylgb.train(x_train, y_train, x_val, y_val)
        mylgb.save(best_model,model_path=os.path.join(root,'model',stock.replace('.csv','.pkl')))
    except:
        print('bad_stock:',stock)
        bad_stocks.append(stock)    

##保存数据或其它方面有问题的stock名
df = pd.DataFrame(bad_stocks, columns=['Stock'])
df.to_csv('data/bad_stock.csv', index=False)