i=df[['EQ', 'Social_Search_Impressions',
       'Digital_Working_cost', 'Print_Working_Cost.Ads50',
       'OOH_Impressions', 'SOS_pct',
       'CCFOT', 'Median_Temp', 'Median_Rainfall',
       'Fuel_Price', 'Inflation',
       'Any_Promo_pct_ACV',
       'Est_ACV_Selling', 'pct_ACV',
       'Avg_no_of_Items', 'pct_PromoMarketDollars_Category',
       'Magazine_Impressions_pct'
       ,'Competitor4_RPI', 'EQ_Category',
       'pct_PromoMarketDollars_Subcategory',
       'Period', 'Year']]


corr = i.corr()
corr.style.background_gradient(cmap='coolwarm')


subdf=corr[['EQ']]
subdf.reset_index(inplace=True)

subdf.to_csv('corr.csv',index=False)


import seaborn as sns
sns.set(style="whitegrid")

ax = sns.barplot(x="index", y="EQ", data=subdf)
