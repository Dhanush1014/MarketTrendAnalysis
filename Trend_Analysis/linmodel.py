#@author: viole
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("flipkart.csv")
X=dataset.iloc[:,6:7]
Y=dataset.iloc[:,7:8]


from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis=0)
imputer.fit(X)
X= imputer.transform(X)
imputer_y= Imputer(missing_values='NaN', strategy = 'mean' , axis=0)
imputer_y.fit(Y)
Y= imputer.transform(Y)
plt.scatter(X,Y)


from sklearn.cross_validation import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size = 0.2, random_state=0)



from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_tr , y_tr)
y_pred = regression.predict(x_te)


plt.scatter(x_tr, y_tr, color = 'red' )
plt.plot(x_tr, regression.predict(x_tr), color = 'blue' )
plt.title('Seasonal Trend vs Price : Training Set')
plt.xlabel('Start of Season Price'); plt.ylabel('End of Season (Discounted) Price')
plt.savefig('templates/retailvsdiscounted.jpg')
plt.show()



plt.scatter(x_te, y_te, color = 'red' )
plt.plot(x_tr, regression.predict(x_tr), color = 'blue' )
plt.title('Seasonal Trend vs Price : Training Set')
plt.xlabel('Start of Season Price'); plt.ylabel('End of Season (Discounted) Price')
plt.savefig('templates/pricevdiscountedtest.jpg')
plt.show()

def getdiscount(price):
    print ("The discounted value for given item : ",regression.predict(price))
getdiscount(5000)

from sklearn.externals import joblib
joblib.dump(regression,'linmodel.pkl')
