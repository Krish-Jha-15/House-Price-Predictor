import pandas as pd
import numpy as np


df=pd.read_csv('./data/Housing_Data-set.csv')
df['House_age']=2025-df['yr_built']
df['Age_of_renovation'] = np.where(df['yr_renovated'] > 0, 2025 - df['yr_renovated'], 0)
df['total_area']=df['sqft_living']+df['sqft_lot']
df['BRB']=df['bedrooms']/df['bathrooms']
df['renovated']=np.where(df.Age_of_renovation>0,1,0)


