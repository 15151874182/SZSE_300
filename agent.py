import baostock as bs
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 登出Baostock系统
bs.logout()

class Agent():
    
    def __init__(self):
        self.seq_len=30 ##月频

    ## 下载沪深300指数，并保存
    def download_300(self,save_dir,start_date,end_date,frequency):
        # 登录Baostock系统
        lg = bs.login()        
        # 获取沪深300指数的历史K线数据
        stock_code = "sh.000300"  # 沪深300指数的代码
        rs = bs.query_history_k_data(stock_code, "date,open,high,low,close,preclose,volume,amount,pctChg", start_date=start_date, end_date=end_date, frequency=frequency)
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 生成DataFrame并保存为CSV文件
        df = pd.DataFrame(data_list, columns=rs.fields)
        file_name="300.csv"
        file_path = os.path.join(save_dir, file_name)
        df.to_csv(file_path, index=False)   
        
        # 登出Baostock系统
        bs.logout()
    
    ## 下载沪深300成分股，并分开保存
    def download_data(self,save_dir,start_date,end_date,frequency):
        # 登录Baostock系统
        lg = bs.login()
        
        # 获取沪深300成分股列表
        rs = bs.query_hs300_stocks()
        hs300_stocks = []
        while (rs.error_code == '0') & rs.next():
            hs300_stocks.append(rs.get_row_data())
            
        # 创建文件夹存储CSV文件
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 循环下载每个成分股数据并保存到CSV文件
        for stock_info in tqdm(hs300_stocks):
            stock_code = stock_info[1]  # 股票代码
            stock_name = stock_info[2]  # 股票名称
            
            # 获取股票历史数据
            rs = bs.query_history_k_data_plus(stock_code, "date,open,high,low,close,preclose,volume,amount,pctChg", start_date=start_date, end_date=end_date, frequency=frequency)
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            # 生成DataFrame并保存为CSV文件
            df = pd.DataFrame(data_list, columns=rs.fields)
            file_name = f"{stock_code}_{stock_name}.csv"
            file_path = os.path.join(save_dir, file_name)
            df.to_csv(file_path, index=False)        

    ## feature_engineering 
    def feature_engineering(self,data):
        
        ####1,2,5,7,10 day lagged returns 
        for i in [1,2,5,7,10]:
            data[f'return_{i}d']=data['return'].shift(i)  
            
        ####momentum,price change over 1,2,5,7,10 days    
        for i in [1,2,5,7,10]:    
            data[f'momentum_{i}d']=data['close']-data['close'].shift(i)   
                
        ####time discretization feature
        data['date']=data.index
        data['DayOfWeek']=data['date'].apply(lambda x : (x.dayofweek-1) / 6.0 - 0.5)
        data['DayOfMonth']=data['date'].apply(lambda x : (x.day - 1) / 30.0 - 0.5)
        data['DayOfYear']=data['date'].apply(lambda x : (x.dayofyear - 1) / 364.0 - 0.5)
        del data['date']
    
        ####SMA(Simple Moving Average)
        for i in [5,7,10,20,30]:
            data[f'SMA_{i}d']=data['close'].rolling(i).mean()
            
        ####EMA(Exponential Moving Average)
        for i in [5,7,10,20,30]:    
            data[f'EMA_{i}d']=data['close'].ewm(5, adjust=False).mean()
        
        ####RSI(Relative Strength Index)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        ####MACD(Moving Average Convergence Divergence)
        # Calculate the short-term and long-term EMAs
        short_ema = data['close'].ewm(span=12, min_periods=1, adjust=False).mean()
        long_ema = data['close'].ewm(span=26, min_periods=1, adjust=False).mean()
        # Calculate the MACD Line
        data['MACD'] = short_ema - long_ema
        # Calculate the Signal Line (usually a 9-period EMA of the MACD Line)
        data['Signal_Line'] = data['MACD'].ewm(span=9, min_periods=1, adjust=False).mean()
    
        ####CCI
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_typical_price = typical_price.rolling(window=5).mean()
        mean_deviation = typical_price.rolling(window=5).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        data['cci'] = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
        
        ####YZ estimator
        n=2 ##The highest efficiency is reached when n=2,a=1.34
        k=0.34/(1.34+(n+1)/(n-1)) ##constant k 
        data['oi']=np.log(data['open']/data['close'].shift(1)) ## the normalized open
        data['ci']=np.log(data['close']/data['open']) ##the normalized close
        data['ui']=np.log(data['high']/data['open']) ##the normalized high
        data['di']=np.log(data['low']/data['open']) ## the normalized low    
        
        o1=data['oi'].shift(1)
        o2=data['oi']
        c1=data['ci'].shift(1)
        c2=data['ci']
        u1=data['ui'].shift(1)
        u2=data['ui']
        d1=data['di'].shift(1)
        d2=data['di']
        
        Vo=0.5*(o1-o2)**2 ##unbias variance of normalized open
        Vc=0.5*(c1-c2)**2 ##unbias variance of normalized close
        VRs=0.5*(u1**2+d1**2-u1*c1-d1*c1+   ##VRs variance found by Rogers and Satchell
                       u2**2+d2**2-u2*c2-d2*c2)
        data['Vyz']=Vo+k*Vc+(1-k)*VRs ##YZ estimator
    
        ####ATR(Average True Range)
        data['H-L']  = abs(data['high']-data['low'])
        data['H-PC'] = abs(data['high']-data['close'].shift(1))
        data['L-PC'] = abs(data['low']-data['close'].shift(1))
        ##function to calculate True Range and Average True Range
        data['TR']   = data[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
        data['ATR']  = data['TR'].rolling(21).mean()
        
        #### OBV(On-Balance Volume)
        data['Price_Change'] = data['close'].diff()
        ##put 'volume' into positive or negtive based on Price_Change
        data['OBV'] = data.apply(lambda row: row['volume'] if row['Price_Change'] > 0 else
                                 (-1) * row['volume'] if row['Price_Change'] < 0 else 0, axis=1)
        data['OBV'] = data['OBV'].cumsum() ##cumulative sum
        del data['Price_Change']
    
        #### VWMA(Volume Weighted Moving Average)
        data['VWMA'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()        
        
        #### Bollinger Bands
        window = 3  # Rolling window size
        num_std = 2  # Number of standard deviations for bands
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        # Calculate upper and lower Bollinger Bands
        data['upper_band'] = rolling_mean + (rolling_std * num_std)
        data['lower_band'] = rolling_mean - (rolling_std * num_std)
        
        ####Small positive returns below 0.25% is labelled as 0 and other values as 1
        data['y']=data['return'].shift(-1*self.seq_len)
        return data
    
    ## feature selected by EDA =====================================================
    def make_dataset(self,FE_data):
        feas_selected=['return_10d', 'close', 'momentum_2d', 'TR', 'Vyz',
                       'VWMA', 'DayOfMonth','DayOfWeek', 'di', 'return_2d', 
                       'DayOfYear', 'cci','ATR', 'MACD','OBV', 'ui', 
                       'volume', 'momentum_7d', 'return_7d','rsi']
        x=FE_data[feas_selected]
        y=FE_data['y']
        x_trainval,x_test,y_trainval,y_test=train_test_split(x,y,shuffle=False,test_size=1/10) ##split test first
        x_train,x_val,y_train,y_val=train_test_split(x_trainval,y_trainval,shuffle=False,test_size=2/9) ##then split train,val         
        return x_train, x_val, x_test, y_train, y_val, y_test


    def get_res(self,pred,gt):
        date_index=gt.index
        pred=np.array(pred).reshape(-1)
        gt=np.array(gt).reshape(-1)
        res=pd.DataFrame(index=date_index)
        res['pred'],res['gt']=pred,gt
        return res

    def eval_res(self,res):
        ## res(DataFrame) must at least have 'gt' and 'pred' cols
        from copy import deepcopy 
        df=deepcopy(res)                            
        error=abs((df['pred']/100+1).prod()-(df['gt']/100+1).prod()) ##return 0.6表示 0.6%，所以要/100
        return error/len(df)