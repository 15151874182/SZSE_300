## import library =============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler ##max-min nomalization
from sklearn.model_selection import train_test_split ##split dataset into train/val/test
import warnings
warnings.filterwarnings('ignore') ##ignore warning

# Set random seeds for reproducibility
seed_value = 123
np.random.seed(seed_value) # 1. Set random seed for numpy
pd.np.random.seed(seed_value) # 2. Set random seed for pandas
timesteps=5 ##example: timestep=5, we will use t1,t2,t3,t4,t5 to predict t6

    
## load_data ====================================================================
def download_data():
    import baostock as bs ## a Python SDK to download stock data, pip install baostock
    import pandas as pd
    lg = bs.login() ##login baostock system
    rs = bs.query_history_k_data_plus("sh.000001",    ###SSEC index code
        "date,open,high,low,close,preclose,volume,amount,pctChg",  ##original features
        start_date='2012-01-01', end_date='2021-01-01', frequency="d")  ###10years data
    data_list = [] ##save every daily data
    while (rs.error_code == '0') & rs.next(): ##go through each daily data
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields) ##change list to dataframe
    
    result.to_csv("./history_SSEC.csv", index=False) ##save datframe to csv
    result.index=pd.to_datetime(result['date']) ##set index to date
    del result['date'] ##delete 'date' column
    bs.logout() ##logout baostock system
    result=result.astype(float)
    return result 
# data=download_data() ##download data from baostock if needed
data=pd.read_csv('history_SSEC.csv',index_col=0,parse_dates=True)
data.rename(columns={'pctChg': 'return'}, inplace=True) 

## feature_engineering ========================================================
def feature_engineering(data):
    
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
    data['y']=data['return'].apply(lambda x: 1 if x>=0.25 else 0).shift(-1*timesteps)
    return data
FE_data=feature_engineering(data)

## data cleaning===================================================================
res=pd.DataFrame()
res['clean_before']=list(FE_data.isnull().sum())
FE_data=FE_data.dropna()
res['clean_after']=list(FE_data.isnull().sum())
res.index=FE_data.columns



## EDA =============================================================================
def som_analysis(data):

    from minisom import MiniSom
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X=X.T
    
    # Initialize a 15x15 SOM
    som = MiniSom(15, 15, X.shape[1], learning_rate=0.5, sigma=5, random_seed=42)
    # Initialize with randon weights
    som.random_weights_init(X)
    # Trains SOM with 10000 iterations
    som.train_batch(X,10000, verbose=True)
    # Plot SOM
    plt.figure(figsize=(20, 10))
    for ix in range(len(X)):
        winner = som.winner(X[ix])
        plt.text(winner[0], winner[1], data.columns[ix], bbox=dict(facecolor='white', alpha=0.5, lw=0)) 
    plt.imshow(som.distance_map())
    plt.colorbar()
    plt.grid(False)
    plt.title('Self Organizing Maps')
    plt.show()
# df2=deepcopy(FE_data)
# del df2['y']
# som_analysis(df2)
## Kmeans Analysis =====================================================
def kmeans_analysis(data):
    from sklearn.cluster import KMeans
    scaler = MinMaxScaler()
    X = scaler.fit_transform(data)
    X=X.T
    
    n_clusters = range(2, 30)
    inertia = []
    
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    plt.plot(n_clusters, np.divide(inertia,inertia[0]))
    plt.hlines(0.1, n_clusters[0], n_clusters[-1], 'r', linestyles='dashed')
    plt.hlines(0.05, n_clusters[0], n_clusters[-1], 'r', linestyles='dashed')
    plt.xlabel('clusters')
    plt.ylabel('relative inertia')
    plt.legend(['inertia', '10% relative inertia', '5% relative inertia'])
    plt.show()
    
    kmeans = KMeans(n_clusters=18)
    kmeans.fit(X)    
    labels =kmeans.predict(X)
    res=pd.DataFrame([data.columns,labels]).T
    res.columns=['feature','cluster']
    sorted_res = res.sort_values(by='cluster')
    print(sorted_res)
# df3=deepcopy(FE_data)
# del df3['y']
# kmeans_analysis(df3)
## DecisionTree Analysis =====================================================
def DT_analysis(data):
    # Separate features and target variable
    y=data['y']
    del data['y'] ##label 'y' removed from features
    X=data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.tree import DecisionTreeClassifier
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    # Create Decision Tree classifier
    clf = DecisionTreeClassifier(random_state=42)
    # Train the classifier on the training data
    clf.fit(X_train, y_train)
    # Get feature importances
    feature_importances = clf.feature_importances_
    # Create a DataFrame to display feature importances
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    print(importance_df)
# df4=deepcopy(FE_data)
# DT_analysis(df4)
##pearson coefficient Analysis =====================================================
def pearson_analysis(data):
    import seaborn as sns
    # Calculate Pearson correlation matrix
    correlation_matrix = data.corr()
    
    # Plot heatmap
    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, fmt='.1f')
    plt.title('Pearson Correlation Heatmap')
    plt.show()    
# df5=deepcopy(FE_data)
# del df5['y']
# pearson_analysis(df5)



## feature selected by EDA =====================================================
def feature_select(data):
    feas_selected=['return_10d', 'close', 'momentum_2d', 'TR', 'Vyz',
                   'VWMA', 'DayOfMonth','DayOfWeek', 'di', 'return_2d', 
                   'DayOfYear', 'cci','ATR', 'MACD','OBV', 'ui', 
                   'volume', 'momentum_7d', 'return_7d','rsi', 'return_10d']
    x=FE_data[feas_selected]
    y=FE_data['y']
    return x,y
x,y=feature_select(FE_data)



## data split and scaling =====================================================
def data_split_scale(x,y):
    scaler = MinMaxScaler()
    x_trainval,x_test,y_trainval,y_test=train_test_split(x,y,shuffle=False,test_size=1/10) ##split test first
    x_train,x_val,y_train,y_val=train_test_split(x_trainval,y_trainval,shuffle=False,test_size=2/9) ##then split train,val 

    x_train_close=x_train['close']
    x_val_close=x_val['close']
    x_test_close=x_test['close']

    plt.figure(figsize=(10, 6))
    plt.plot(x_train_close, 'b-', label='Training Set')
    plt.plot(x_val_close, 'g-', label='Validation Set')
    plt.plot(x_test_close, 'r-', label='Testing Set')
    
    # add axvline
    plt.axvline(x_train_close.index[-1], color='k', linestyle='--', label='Train/Validation Split')
    plt.axvline(x_val_close.index[-1], color='k', linestyle='--', label='Validation/Testing Split')

    plt.legend()
    plt.title("Dataset Split Visualization")
    plt.xlabel("Date")
    plt.ylabel("Close price")

    plt.tight_layout()
    plt.show()        
    plt.close()
    
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    print('x_train:',x_train.shape)
    print('x_val:',x_val.shape)
    print('x_test:',x_test.shape)
    print('y_train:',y_train.shape)
    print('y_val:',y_val.shape)
    print('y_test:',y_test.shape)
    
    return x_train, x_val, x_test, y_train, y_val, y_test, x_train_scaled,x_val_scaled,x_test_scaled
x_train, x_val, x_test, y_train, y_val, y_test, x_train_scaled,x_val_scaled,x_test_scaled=data_split_scale(x,y)



## subset selection method
def filter_method(x_train,y_train):
    from sklearn.feature_selection import SelectKBest,f_classif
    #### Use correlation coefficients to select most relevant features
    selector = SelectKBest(score_func=f_classif, k=int(len(x_train.columns)*0.8))  ## we can see k equals to 80% original columns length      
    x_selected = selector.fit_transform(x_train, y_train)
    feature_indices = selector.get_support(indices=True)
    filter_selected_feas = x_train.columns[feature_indices]
    print('filter_selected_feas:',filter_selected_feas)
    print('filter_selected_feas_number:',len(filter_selected_feas))
    return filter_selected_feas,feature_indices
filter_selected_feas,filter_indices=filter_method(x_train,y_train)

def wrapper_method(x_train,y_train):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import RFE
    base_model = DecisionTreeClassifier(random_state=42)
    rfe = RFE(estimator=base_model, n_features_to_select=15)
    rfe.fit(x_train, y_train)
    wrapper_selected_feas = x_train.columns[rfe.support_]
    feature_indices = np.where(rfe.support_)[0]
    print('wrapper_selected_feas:',wrapper_selected_feas)
    print('wrapper_selected_feas_number:',len(wrapper_selected_feas))
    return wrapper_selected_feas,feature_indices
wrapper_selected_feas,wrapper_indices=wrapper_method(x_train,y_train)

def embedded_method(x_train,y_train):
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=0.001)  
    lasso.fit(x_train,y_train)
    embedded_selected_feas = x_train.columns[lasso.coef_ != 0]
    feature_indices = np.where(lasso.coef_ != 0)[0]
    print('embedded_selected_feas:',embedded_selected_feas)
    print('embedded_selected_feas_number:',len(embedded_selected_feas))
    return embedded_selected_feas,feature_indices
embedded_selected_feas,embedded_indices=embedded_method(x_train,y_train)


## create_lstm_dataset =====================================================
##the x_train,x_val,x_test are 3-D as (samples,timestep,feature_num)
##the y_train,y_val,y_test are 2-D as (samples,1)
##the [:,0,:] means get the first element as label
def create_lstm_dataset(data):
    stack = []
    for i in range(0, len(data) - timesteps + 1):
        window = data[i:i+timesteps,:]
        stack.append(window)
    return np.array(stack)
x_train_scaled=create_lstm_dataset(x_train_scaled)
y_train=create_lstm_dataset(np.array(y_train).reshape(-1,1))[:,0,:]
x_val_scaled=create_lstm_dataset(x_val_scaled)
y_val=create_lstm_dataset(np.array(y_val).reshape(-1,1))[:,0,:]
x_test_scaled=create_lstm_dataset(x_test_scaled)
y_test=create_lstm_dataset(np.array(y_test).reshape(-1,1))[:,0,:]
print('x_train:',x_train_scaled.shape)
print('x_val:',x_val_scaled.shape)
print('x_test:',x_test_scaled.shape)
print('y_train:',y_train.shape)
print('y_val:',y_val.shape)
print('y_test:',y_test.shape)


import tensorflow as tf
tf.random.set_seed(seed_value) #Set random seed for tensorflow/keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.losses import binary_crossentropy

## create keras model =====================================================
##baseline-model with 1 LSTM layer( 32 units) and 1 Dense layers
# base_model = Sequential(name="base_model")
# base_model.add(LSTM(32, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# base_model.add(Dropout(0.1))
# base_model.add(Dense(1, activation='sigmoid'))
# base_model.summary()

# ##model with 1 LSTM layer( 64 units) and 1 Dense layers
# model2 = Sequential(name="model2")
# model2.add(LSTM(64, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# model2.add(Dropout(0.1))
# model2.add(Dense(1, activation='sigmoid'))
# model2.summary()

# ##model with 1 LSTM layer( 128 units) and 1 Dense layers
# model3 = Sequential(name="model3")
# model3.add(LSTM(128, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# model3.add(Dropout(0.1))
# model3.add(Dense(1, activation='sigmoid'))
# model3.summary()

# ##model with 1 LSTM layer( 32 units) and 2 Dense layers
# model4 = Sequential(name="model4")
# model4.add(LSTM(32, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# model4.add(Dropout(0.1))
# model4.add(Dense(16, activation='tanh'))
# model4.add(Dense(1, activation='sigmoid'))
# model4.summary()

# ##model with 1 LSTM layer( 64 units) and 2 Dense layers
# model5 = Sequential(name="model5")
# model5.add(LSTM(64, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# model5.add(Dropout(0.1))
# model5.add(Dense(16, activation='tanh'))
# model5.add(Dense(1, activation='sigmoid'))
# model5.summary()

# ##model with 1 LSTM layer( 128 units) and 2 Dense layers
# model6 = Sequential(name="model6")
# model6.add(LSTM(128, activation='tanh',input_shape=(timesteps, x_train_scaled.shape[2]), return_sequences=False))
# model6.add(Dropout(0.1))
# model6.add(Dense(16, activation='tanh'))
# model6.add(Dense(1, activation='sigmoid'))
# model6.summary()

# ##model with 1 LSTM layer( 32 units) and 1 Dense layers,input_shape use filter_indices
# model7 = Sequential(name="model7")
# model7.add(LSTM(32, activation='tanh',input_shape=(timesteps, len(filter_indices)), return_sequences=False))
# model7.add(Dropout(0.1))
# model7.add(Dense(1, activation='sigmoid'))
# model7.summary()

# ##model with 1 LSTM layer( 32 units) and 1 Dense layers,input_shape use filter_indices
# model8 = Sequential(name="model8")
# model8.add(LSTM(32, activation='tanh',input_shape=(timesteps, len(wrapper_indices)), return_sequences=False))
# model8.add(Dropout(0.1))
# model8.add(Dense(1, activation='sigmoid'))
# model8.summary()

# ##model with 1 LSTM layer( 32 units) and 1 Dense layers,input_shape use filter_indices
# model9 = Sequential(name="model9")
# model9.add(LSTM(32, activation='tanh',input_shape=(timesteps, len(embedded_indices)), return_sequences=False))
# model9.add(Dropout(0.1))
# model9.add(Dense(1, activation='sigmoid'))
# model9.summary()

from keras.optimizers import SGD
optimizer = SGD(learning_rate=0.01)
# Calculate sample weights based on class labels
class_counts = np.bincount(y_train.reshape(-1).astype(int)) ##calculate numbers of 2 classes
total = len(y_train) ##total number of sample
class_weight = {0: total / (2 * class_counts[0]), 1: total / (2 * class_counts[1])}


def train_evaluate_model(model,class_weight,batch_size,x_train, x_val, x_test, y_train, y_val, y_test):
    # Compile the model with binary_crossentropy and binary_accuracy
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['binary_accuracy'])
    from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping
    tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=100, write_graph=True, write_images=True)
    ## tensorboard --logdir=./logs
    # Set the ModelCheckpoint callback function to save the model with the highest verification accuracy
    checkpoint = ModelCheckpoint('best_model.h5', monitor='binary_accuracy', save_best_only=True, mode='max', verbose=1)
    
#     early_stopping = EarlyStopping(
#     monitor='val_auc',  # monitor indicator
#     patience=400,  # Stop training when the neutral energy has not improved for 200 epochs in a row
#     restore_best_weights=True  # Restore optimal weight
#     )
    history = model.fit(x_train, y_train, epochs=1500, batch_size=batch_size, verbose=1,
                        validation_data=(x_val, y_val), 
                        callbacks=[checkpoint,tensorboard_callback],class_weight=class_weight)
    
    # plot loss and accuracy diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['binary_accuracy'], label='train')
    ax2.plot(history.history['val_binary_accuracy'], label='validation')
    ax2.set_title('Model binary_accuracy')
    ax2.set_ylabel('binary_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

    ##predict_model =====================================================
    from keras.models import load_model
    y_test_pred = model.predict(x_test) 
    y_test_pred_binaray = (y_test_pred > 0.5).astype(int)##binary output
    ##plot confusion_matrix =====================================================
    def plot_confusion_matrix(y_test,y_test_pred):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(y_test.reshape(-1), y_test_pred.reshape(-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
        disp.plot(cmap=plt.cm.Blues, values_format='d')
        plt.show()
    plot_confusion_matrix(y_test,y_test_pred_binaray)
    
    ##plot classification_report =====================================================
    from sklearn.metrics import classification_report
    print(classification_report(y_test.reshape(-1), y_test_pred_binaray.reshape(-1)))
    
    ##plot roc_curve =====================================================
    def plot_roc_curve(y_test,y_test_pred):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds  = roc_curve(y_test.reshape(-1), y_test_pred.reshape(-1))
        # 计算约登指数
        youden_index = tpr + (1 - fpr) - 1
        # 找到最大约登指数的索引
        best_threshold_index = np.argmax(youden_index)
        # 最佳阈值
        best_threshold = thresholds[best_threshold_index]
        print("Best Threshold:", best_threshold)
        
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()
    plot_roc_curve(y_test,y_test_pred)



##model with 1 LSTM layer( 32 units) and 1 Dense layers,input_shape use filter_indices
model7 = Sequential(name="model7")
model7.add(LSTM(32, activation='tanh',input_shape=(timesteps, len(filter_indices)), return_sequences=False))
model7.add(Dropout(0.1))
model7.add(Dense(1, activation='sigmoid'))
## test three different subset features: 16 features selected through the filter method
train_evaluate_model(model=model7,
                      class_weight=None,
                      batch_size=64,
                      x_train=x_train_scaled[:, :,filter_indices], x_val=x_val_scaled[:, :,filter_indices], x_test=x_test_scaled[:, :,filter_indices],
                      y_train=y_train, y_val=y_val, y_test=y_test)



y_test_pred = model7.predict(x_test_scaled[:, :,filter_indices]) 
y_test_pred_binaray = (y_test_pred > 0.5).astype(int)##binary output
asset_prices = x_test['close'][5:] ##get prediction date close price
y_test_pred_binaray=y_test_pred_binaray[:-1] ##we dont know the last prediction date close price


def backtesting(asset_prices,y_test_pred_binaray):
    # Transaction cost (0.15%)
    transaction_cost = 0.0015
    # Initial portfolio value (units)
    initial_portfolio = 1.0
    portfolio_value = initial_portfolio
    position = 0  # 0: No position, 1: Long position
    trades = []
    portfolio_values = [portfolio_value]
    # Loop through each time step
    for i in range(len(y_test_pred_binaray)):
        signal = y_test_pred_binaray[i]
        price = asset_prices[i]
        
        if signal == 1 and position == 0:  # Buy signal and no position
            position = portfolio_value / price
            portfolio_value -= portfolio_value * transaction_cost
            trades.append((i, "Buy", price))
        elif signal == 0 and position > 0:  # Sell signal and existing position
            portfolio_value = position * price
            portfolio_value -= portfolio_value * transaction_cost
            position = 0
            trades.append((i, "Sell", price))
        
        portfolio_values.append(portfolio_value)
    
    # Handle remaining position at the end
    if position > 0:
        portfolio_value = position * asset_prices[-1]
    
    # Calculate returns
    total_return = (portfolio_value - initial_portfolio) / initial_portfolio
    
    print("Total Return:", total_return)
    print("Trades:", trades)
    
    # Visualize portfolio value over time
    plt.plot(asset_prices.index[1:], portfolio_values, label="Portfolio Value")
    plt.xlabel("date")
    plt.xticks(rotation=90)
    plt.ylabel("Portfolio Value")
    plt.title("Portfolio Value Over Time")
    plt.legend()
    plt.show()

backtesting(asset_prices,y_test_pred_binaray)
















# # Transaction cost (0.15%)
# transaction_cost = 0.0015
# # Initial portfolio value (units)
# initial_portfolio = 1.0
# portfolio_value = initial_portfolio
# position = 0  # 0: No position, 1: Long position
# trades = []
# # Loop through each time step
# for i in range(len(y_test_pred_binaray)):
#     signal = y_test_pred_binaray[i]
#     price = asset_prices[i]
    
#     if signal == 1 and position == 0:  # Buy signal and no position
#         position = portfolio_value / price
#         portfolio_value -= portfolio_value * transaction_cost
#         trades.append((i, "Buy", price))
#     elif signal == 0 and position > 0:  # Sell signal and existing position
#         portfolio_value = position * price
#         portfolio_value -= portfolio_value * transaction_cost
#         position = 0
#         trades.append((i, "Sell", price))

# # Handle remaining position at the end
# if position > 0:
#     portfolio_value = position * asset_prices[-1]

# # Calculate returns
# total_return = (portfolio_value - initial_portfolio) / initial_portfolio

# print("Total Return:", total_return)
# print("Trades:", trades)