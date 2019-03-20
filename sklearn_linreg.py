#%%
'''This notebook explores linear regression in sklearn'''
import pandas as pd 
import numpy as np 
import sklearn
from sklearn import linear_model, model_selection
from sklearn.model_selection import train_test_split
import requests as req
import matplotlib.pyplot as plt
#%%
'''Simple (single independent variable) linear regression'''
dataset = pd.read_csv ('/users/richardkornblith/downloads/student_scores.csv')
dataset.info()
dataset.plot(x='Hours', y = 'Scores', style='o')
plt.title ('Hours Studied vs. Percentage Score')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
X = dataset.iloc[:,:-1].values
print(type(X))
y = dataset.iloc[:,1].values
print(type(y))
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size = 0.20, random_state = 0)
print(X_train.shape, X_test.shape)
from sklearn.linear_model import LinearRegression
lmod = LinearRegression()
lmod.fit(X_train, y_train)
print(lmod.coef_, lmod.intercept_)
y_predlm = lmod.predict(X_test)
from sklearn import metrics
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_predlm))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test, y_predlm))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_predlm)))



#%%
'''Multiple linear regression'''
petrodata = pd.read_csv('/users/richardkornblith/downloads/petrol_consumption.csv')
print(petrodata.head())
print(petrodata.info())
print(petrodata.describe())
print(petrodata.columns)
X_multi = petrodata[['Petrol_tax','Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']]
y_multi = petrodata['Petrol_Consumption']
X_multi.head()
print('y_multi.head()', y_multi.head())

#%%
X_train, X_test, y_train, y_testmult = train_test_split(X_multi, y_multi, test_size = .20, random_state = 0)
multLR = LinearRegression()
multLR.fit(X_train, y_train)
coeff_df = pd.DataFrame(multLR.coef_)#, X.columns)#, columns=['Coefficient'])
coeff_df
y_predmult = multLR.predict(X_test)
df = pd.DataFrame({'Actual': y_testmult, 'Predicted': y_predmult})
df
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_testmult, y_predmult))
print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_testmult, y_predmult))
print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_testmult, y_predmult)))
#%%
