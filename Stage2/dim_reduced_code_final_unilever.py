import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
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

import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV



tr=pd.read_excel('TrainingData.xlsx')
tr.shape


te=pd.read_excel('TestData.xlsx')
te.shape

df=tr.append(te)

def fun2(str1):
    return(str1.split(':')[1]) 

df['Period']=df['Period'].apply(fun2)

df_tr=df.iloc[0:34,:]
df_te=df.iloc[34:,:]

df1=df_tr.copy()
df2=df_tr.copy()


pr1=df1.groupby('Period').agg({'EQ': 'sum'})
pr2=df1.groupby('Period').agg({'EQ': 'min'})
pr3=df1.groupby('Period').agg({'EQ': 'max'})
pr4=df1.groupby('Period').agg({'EQ': 'mean'})
pr5=df1.groupby('Period').agg({'EQ': 'var'})

pr_encoding=pd.concat([pr1,pr2,pr3,pr4,pr5], axis=1)
pr_encoding.reset_index(inplace=True)
pr_encoding.columns=['Period','period_sum_eq','period_min_eq','period_max_eq','period_mean_eq','period_var_eq']


df3=pd.merge(df1,pr_encoding,on='Period',how='inner')
df4=pd.merge(df_te,pr_encoding,on='Period',how='inner')
del df4['Period']
del df3['Period']

f=df3.corr()


df3=df3.fillna(method='ffill')
df3=df3.fillna(method='bfill')
df4=df4.fillna(method='ffill')
df4=df4.fillna(method='bfill')
####################################################
##mean and median encoding for train data for year and period


X =df3[['Social_Search_Impressions', 'Social_Search_Working_cost',
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
       'RPI_Subcategory', 'period_sum_eq', 'period_min_eq', 'period_max_eq',
       'period_mean_eq', 'period_var_eq']]
y = df3['EQ']
##############################################################
X_te =df4[['Social_Search_Impressions', 'Social_Search_Working_cost',
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
       'RPI_Subcategory', 'period_sum_eq', 'period_min_eq', 'period_max_eq',
       'period_mean_eq', 'period_var_eq']]
y_te = df4['EQ']
###################################################################

train_X, test_X, train_y, test_y = train_test_split( X,y,test_size = 0.15,
                                                    random_state = 49)



def results_test(test_x,test_y,model,model_name):
    y_pred_tr = model.predict(test_x)
    df2 = pd.DataFrame(y_pred_tr,test_y).reset_index()
    df2.columns = ['y_test', 'y_pred']
    #df2.to_csv('test_pred.csv')
    kk = pd.DataFrame(df2.describe()).reset_index()
    kk['diff'] = abs(kk['y_test'] -kk['y_pred'])/kk['y_test']
    mape = np.mean(np.abs((df2['y_test'] - df2['y_pred']) / df2['y_test'])) * 100
    rmse = np.sqrt(mean_squared_error(df2['y_test'], df2['y_pred']))
    mae = np.mean(np.abs(df2['y_test'] - df2['y_pred']))
    print('MAPE:',mape,'RMSE:',rmse,'MAE:',mae)

def cv_xgb(train_X,train_y):
    model = xgb.XGBRegressor()
    param_dist = {"max_depth": [3,5,7,9],
              "min_child_weight" : [1,3,6,10,20],
              "n_estimators": [30,100,120,150,200],
              "learning_rate": [0.05, 0.1,0.01,0.001],}
    grid_search = GridSearchCV(model,scoring='neg_mean_absolute_error',
                           param_grid=param_dist, cv = 5, 
                                   verbose=10, n_jobs=-1)
    grid_search.fit(train_X,train_y)
    print(grid_search.best_estimator_)
    print(grid_search.best_score_)
    final_xg=grid_search.best_estimator_
    final_xg.fit(train_X,train_y)
    return final_xg

dim_red_xg=cv_xgb(train_X, train_y)




def feature_impotance(final_xg):
    feat_imp=final_xg.get_booster().get_score(importance_type="gain")
    feat_imp=pd.DataFrame(feat_imp.items())
    feat_imp.columns=['features','importance']
    feat_imp.sort_values('importance',inplace=True,ascending=False)
    feat_imp.reset_index(drop=True,inplace=True)
    feat_imp=feat_imp.loc[feat_imp['importance']>=200]
    ##as part of dimension reduction using this for reduction 
    ##p =21 and n=34
    feat_imp.to_excel('feat_importance_dimension_reduction_gain.xlsx',index=False)
    return feat_imp

results_test(train_X,train_y,dim_red_xg,'xgb')
results_test(test_X,test_y,dim_red_xg,'xgb')
results_test(X_te,y_te,dim_red_xg,'xgb')
feat_imp=feature_impotance(dim_red_xg)

final_features=list(feat_imp.features)

train_X1=train_X[final_features]
train_y1=train_y
test_X1=test_X[final_features]
test_y1=test_y
X_te1=X_te[final_features]
y_te1=y_te


final_xgb=cv_xgb(train_X1,train_y1)
print('Train results>>>>>>>>>>>>')
results_test(train_X1,train_y1,final_xgb,'xgb')
print('Validation Results>>>>>>>>>>')
results_test(test_X1,test_y1,final_xgb,'xgb')
print('Test Results>>>>>>>>>>')
results_test(X_te1,y_te1,final_xgb,'xgb')

test_results={'EQ':y_te,'Predicted_Eq':final_xgb.predict(X_te1)}
test_results=pd.DataFrame(test_results)
test_results.to_excel('final_actual_vs_pred_test.xlsx',index=False)


