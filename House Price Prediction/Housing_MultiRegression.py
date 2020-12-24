import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

df=pd.read_csv("housing.csv")

df.fillna(0, inplace = True)

x=df.iloc[:,:8].values
y=df.iloc[:,8].values

X=(x-x.mean())/(x.max()-x.min())

# m,n=x.shape

x_train,x_test,y_train,y_test=train_test_split(X,y, test_size= 0.05, random_state=3)

y_train=np.array([y_train])
y_test=np.array([y_test])

reg=linear_model.LinearRegression()
reg.fit(x_train,y_train.T)
y_pred=reg.predict(x_test)

from sklearn.metrics import r2_score
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_pred - y_test.T)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_pred - y_test.T) ** 2))
print("R2-score: %.2f" % r2_score(y_pred , y_test.T) )
