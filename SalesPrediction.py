import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import  make_scorer 
import numpy as np
from random import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score 
from sklearn.preprocessing import MinMaxScaler
import math 

import lightgbm as lgb 

#As this is day wise data aggreting on 28 days for getting period #
tr=pd.read_excel('Training-Data-Sets.xlsx')
df=tr.copy()

df['period flag']=0
count =1


for i in  range(df.shape[0]):
    if (df.iloc[i,0] % 28)!= 0:
        print(i)
        df.iloc[i,39]=count
    else:
        df.iloc[i,39]=count
        count=count+1

del df['Day']

u=pd.DataFrame(df['period flag'].value_counts())
grp_df=df.groupby(['period flag']).mean()
grp_df.reset_index(inplace=True)

grp_df['year']=np.nan

year=1983 ##decided from test dataset based on no of days pased
for i in  range(grp_df.shape[0]):
    if (grp_df.iloc[i,0] % 13)!= 0:
        print(i)
        grp_df.iloc[i,39]=year
    else:
        grp_df.iloc[i,39]=year
        year=year+1


grp_df['Period']=grp_df['period flag'] % 13
grp_df['Period']=np.where(grp_df['Period']==0,13,grp_df['Period'])
grp_df['Year']=grp_df['year']
del grp_df['period flag']
del grp_df['year']


tr=grp_df.copy()
te=pd.read_excel('Test dataset v1.xlsx')

def fun1(str1):
    return(str1.split(' - ')[0])

def fun2(str1):
    return(str1.split(':')[1]) 

te['Year']=te['Period'].apply(fun1)
te['Period']=te['Period'].apply(fun2)


df=pd.concat([tr,te])
df.columns
df['Period']=df['Period'].astype('int32')
df['Year']=df['Year'].astype('int32')
r=df.corr()




X =df[['Social_Search_Impressions', 'Social_Search_Working_cost',
       'Digital_Impressions', 'Digital_Working_cost',
       'Print_Impressions.Ads40', 'Print_Working_Cost.Ads50',
       'OOH_Impressions', 'OOH_Working_Cost', 'SOS_pct',
       'Digital_Impressions_pct', 'CCFOT', 'Median_Temp', 'Median_Rainfall',
       'Fuel_Price', 'Inflation', 'Trade_Invest', 'Brand_Equity',
       'Avg_EQ_Price', 'Any_Promo_pct_ACV', 'Any_Feat_pct_ACV',
       'Any_Disp_pct_ACV', 'EQ_Base_Price', 'Est_ACV_Selling', 'pct_ACV',
       'Avg_no_of_Items', 'pct_PromoMarketDollars_Category', 'RPI_Category',
       'Magazine_Impressions_pct', 'TV_GRP', 'Competitor1_RPI',
       'Competitor2_RPI', 'Competitor3_RPI', 'Competitor4_RPI', 'EQ_Category',
       'EQ_Subcategory', 'pct_PromoMarketDollars_Subcategory',
       'RPI_Subcategory', 'Period', 'Year']]
y = df['EQ']

scaler = MinMaxScaler()
X=scaler.fit_transform(X)

X_tr = X[0:429]
X_te = X[429:]
y_tr = y[0:429]
y_te = y[429:]

train_X, test_X, train_y, test_y = train_test_split( X_tr,y_tr,test_size = 0.2,
                                                    random_state = 49)




def R2(y_true,y_pred):    
     r2 = r2_score(y_true, y_pred)
     return r2
 
R2scorer=make_scorer(R2,greater_is_better=True)    



random_grid1 = {'loss':['lad', 'huber'],
               'n_estimators': [100,200,500,1000],
               'max_depth': [3,4,5,7,9],
               'alpha': [0.7,0.8,0.9],
               'subsample':[0.7,0.8,0.9],
               'max_features':[2,3,4,6,8,15,20],
               'learning_rate':[0.1,0.05]
              }     
 
def cv_train_test(params,train_X,train_y):
    
    ml = GradientBoostingRegressor()
    ml_random = RandomizedSearchCV(estimator = ml,scoring='neg_root_mean_squared_error', param_distributions = params, 
                                   n_iter = 25, 
                               cv = 5,verbose=10, random_state=99, n_jobs = -1)
    ml_random.fit(train_X, train_y)
    print(ml_random.best_score_)
    print(ml_random.best_estimator_)
    ml_final=ml_random.best_estimator_
    ml_final=ml_final.fit(train_X,train_y)
    return ml_final
    
final_gbm=cv_train_test(random_grid1,train_X,train_y)

random_grid2 = {
               'n_estimators': [200,500,100,1200,1500],
               'max_features': [20,8,9,10],
               'max_depth': [3,4,5,7,9],
               'min_samples_split': [1,2, 5, 10,50,100]
               }

def cv_train_test(params,train_X,train_y):
    #train_X, test_X, train_y, test_y = train_test_split( X,y,test_size = 0.2,random_state = randint(0, 100))
    ml = RandomForestRegressor()
    ml_random = RandomizedSearchCV(estimator = ml,scoring='neg_root_mean_squared_error', param_distributions = params, 
                                   n_iter = 25, 
                               cv = 5,verbose=10, random_state=randint(0, 100), n_jobs = -1)
    ml_random.fit(train_X, train_y)
    print(ml_random.best_estimator_)
    print(ml_random.best_score_)
    ml_final=ml_random.best_estimator_
    ml_final=ml_final.fit(train_X,train_y)
    return ml_final

final_rfr=cv_train_test(random_grid2,train_X,train_y)


random_grid3={
     'hidden_layer_sizes': [(50,25),(100,),(50,),(25,),(10,),(5,),(100,),(200)],
     'activation': ['tanh', 'relu'],
     'solver': ['sgd', 'adam'],
     'alpha': [0.01, 0.05,0.001],
     'learning_rate': ['constant','adaptive'],
}


def cv_train_test(params,train_X,train_y):
    mlp = MLPRegressor(max_iter=10000,validation_fraction=0.1)
    ml_random = RandomizedSearchCV(estimator = mlp,scoring='neg_root_mean_squared_error',
                                   param_distributions = params, n_iter = 20, 
                               cv = 5,verbose=10, random_state=randint(0, 100), n_jobs = -1)
    ml_random.fit(train_X, train_y)
    print(ml_random.best_estimator_)
    print(ml_random.best_score_)
    ml_final=ml_random.best_estimator_
    ml_final=ml_final.fit(train_X,train_y)
    return ml_final
    
final_mlp=cv_train_test(random_grid3,train_X,train_y)

def results_test(test_x,test_y,model,model_name):
    y_pred_tr = model.predict(test_x)
    df2 = pd.DataFrame(y_pred_tr,test_y).reset_index()
    df2.columns = ['y_test', 'y_pred']
    df2.to_csv('test_pred.csv')
    kk = pd.DataFrame(df2.describe()).reset_index()
    kk['diff'] = abs(kk['y_test'] -kk['y_pred'])/kk['y_test']
    mape = np.mean(np.abs((df2['y_test'] - df2['y_pred']) / df2['y_test'])) * 100
    rmse = np.sqrt(mean_squared_error(df2['y_test'], df2['y_pred']))
    mae = np.mean(np.abs(df2['y_test'] - df2['y_pred']))
    print('MAPE:',mape,'RMSE:',rmse,'MAE:',mae)


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 100,
        "learning_rate" : 0.001,
        "bagging_fraction" : 0.8,
        "feature_fraction" : 0.8,
        "bagging_frequency" : 10,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "seed": 42
    }
    
    lgtrain = lgb.Dataset(train_X, label= train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain,10000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, 
                                         num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result    
    
pred_test, final_lgb, evals_result = run_lgb(train_X, train_y, test_X, test_y,X_te)
print("LightGBM Training Completed...")


print("Features Importance...")
gain = final_lgb.feature_importance('gain')
featureimp = pd.DataFrame({'feature':final_lgb.feature_name(), 
                   'split':final_lgb.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:20])

print('Results train  and validation MLP>>>>>>>')
results_test(train_X,train_y,final_mlp,',mlp')
results_test(test_X,test_y,final_mlp,',mlp')


print('Results train  and validation RFR>>>>>')
results_test(train_X,train_y,final_rfr,',rfr')
results_test(test_X,test_y,final_rfr,',rfr')

print('Results train  and validation GBM>>>>>')
results_test(train_X,train_y,final_gbm,',gbm')
results_test(test_X,test_y,final_gbm,'gbm')

print('Results train  and validation LGB>>>>')
results_test(train_X,train_y,final_lgb,'lgb') 
results_test(test_X,test_y,final_lgb,'lgb') 








pred_mlp=final_mlp.predict(test_X)
pred_lgb=final_lgb.predict(test_X)        
pred=0.7*pred_lgb+0.3*pred_mlp
mape = np.mean(np.abs(test_y - pred) /test_y ) * 100
print('mape on validation data after esembling',mape)


pred_mlp=final_mlp.predict(X_te)
pred_lgb=final_lgb.predict(X_te)        
pred=0.65*pred_lgb+0.35*pred_mlp
mape = np.mean(np.abs(y_te - pred) /y_te ) * 100
print('mape on test data after ensembling',mape)

final_df={'EQ':list(y_te),'pred_EQ':list(pred)}
final_df=pd.DataFrame.from_dict(final_df)
final_df.to_csv('predictions_test_set',index=False)
