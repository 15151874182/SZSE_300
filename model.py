# -*- coding: utf-8 -*-
import pandas as pd
import warnings
import os
import joblib    
warnings.filterwarnings('ignore') 

import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]

class MyLGB():
    def __init__(self):
        pass
    
    def build_model(self):
        return lgb.LGBMRegressor()  
        # return lgb.LGBMRegressor(**self.config.lgb_param)  
    def train(self, x_train, y_train,x_val, y_val):
        model = self.build_model()
        model.fit(x_train,
                  y_train,
                  eval_set=[(x_val, y_val)],
                  # callbacks=callbacks
                  )
        return model
    
    def predict(self, model,x_test):
        y_pred = model.predict(x_test)
        y_pred=pd.DataFrame(y_pred,index=x_test.index,columns=['pred']) ##put date info into datframe index
        return y_pred

    def save(self, model,model_path):
        # model_save===========================================================        
        joblib.dump(model, model_path)
        
    def load(self, model_path):
        # model load===========================================================    
        best_model = joblib.load(model_path)
        return best_model
        
    def finetune(self, config, x_train,y_train,x_val,y_val, n_trials=100):

        import optuna

        def objective(trial):
            
            # Define hyperparameter Search Scope
            param = {
                'boosting_type':'gbdt',
                'class_weight':None, 
                'colsample_bytree':1.0, 
                'device':'cpu',
                'importance_type':'split', 
                'learning_rate':trial.suggest_float('learning_rate', 1e-5,1e-1),
                'max_depth':trial.suggest_int('max_depth', 2,10,step=1),
                'min_child_samples':91, 
                'min_child_weight':0.001,
                'min_split_gain':0.2, 
                'n_estimators':trial.suggest_int('n_estimators', 50,300,step=10),
                'n_jobs':-1, 
                'num_leaves':trial.suggest_int('max_depth', 2,50,step=1),
                'objective':None, 
                'random_state':1822, 
                'reg_alpha':trial.suggest_float('reg_alpha', 0.1, 1,step=0.1),
                'reg_lambda':trial.suggest_float('reg_lambda', 0.1, 1,step=0.1),
                'silent':True, 
                'subsample':trial.suggest_float('subsample', 0.1, 1,step=0.1), 
                'subsample_for_bin':200000,
                'subsample_freq':0
            }

            model = lgb.LGBMRegressor(**param)
            model.fit(x_train,y_train)
            y_val_pred = model.predict(x_val)
            from model.tools.get_res import get_res
            from model.tools.eval_res import eval_res
            res=get_res(y_val_pred, y_val)
            acc_mape=eval_res(res, config.capacity)            

            return acc_mape
        
        study = optuna.create_study(
            direction='maximize')  # maximize the auc
        study.optimize(objective, n_trials=n_trials)
        print("Best parameters:", study.best_params)
        best_model = lgb.LGBMRegressor( **study.best_params)
        best_model.fit(x_train,y_train)
        return best_model        