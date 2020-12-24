import pandas as pd # Data Set Import...
import matplotlib.pyplot as plt #Graph PLot
import numpy as np #...
from sklearn import linear_model   #Models
from sklearn.model_selection import train_test_split        #Data Spliting

Std_Data=pd.read_csv("student-por.csv", usecols=['G1','G2','G3'])


# print(Std_Data.head())

x=Std_Data[['G1','G2']].values
y=Std_Data['G3'].values

y=np.array([y])
y=y.T


Model=linear_model.LinearRegression()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.05,random_state=42)

Model.fit(x_train,y_train)

print("Coeff: ",Model.coef_)
print('Inercept: ',Model.intercept_)

y_pred=Model.predict(x_test)


from sklearn import metrics
print("Mean Valued Error: ",metrics.mean_squared_error(y_test,y_pred))
print("R2 Value:",metrics.r2_score(y_test,y_pred))

plt.figure(figsize=(4,4))
plt.scatter(x_train[:,1],y_train)
# plt.scatter(x_test[:,1],y_test,color='blue')
# plt.scatter(x_test[:,1],y_pred,color='green')
plt.plot(x_train[:,1],Model.coef_[0][1]*x_train[:,1]+Model.intercept_[0],color='red')
plt.title("Linear Regression")
plt.xlabel("G2")
plt.ylabel('G3')
plt.show()

Std_Data1=Std_Data.head(10)
Std_Data1.plot(kind='bar',figsize=(4,4))
plt.title("Bar Chart")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()