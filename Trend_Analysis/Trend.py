import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv("shirt.csv")
X=dataset.iloc[:,2:6].values
Y=dataset.iloc[:,6:7].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
X[:,0] = labelencoder_x.fit_transform(X[:,0])
ohe = OneHotEncoder(categorical_features=[0])
X = ohe.fit_transform(X).toarray()
X=X[:,1:]

from sklearn.cross_validation import train_test_split
x_tr, x_te, y_tr, y_te = train_test_split( X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regression1= LinearRegression()
regression1.fit(x_tr, y_tr)

y_predict = regression1.predict(x_te)
plt.scatter(X[:,3],Y[:,-1],color='red')
plt.plot(x_tr, regression1.predict(x_tr), color = 'blue' )
plt.xlabel('Price');plt.ylabel('Year');plt.title('Average model line fit')
plt.savefig('static/pricevsyear')
plt.show()

plt.scatter(X[:,4],Y[:,-1],color='yellow')
plt.scatter(X[:,5],Y[:,-1],color='black')
plt.plot(x_tr, regression1.predict(x_tr), color = 'pink')
plt.xlabel('Min Pricing for the item');plt.ylabel('Year based Seasonal Price')
plt.savefig('static/minvsyear')
plt.show()

plt.scatter(X[:,5],Y[:,-1],color='gold')
plt.xlabel('Seasonal Price Inflation %');plt.ylabel('Season : Year')
plt.title('Inflation vs Year')
plt.savefig('static/inflationvsyear')
plt.show()


plt.plot(x_tr, regression1.predict(x_tr), color = 'green' )
plt.title('Item Prices')
plt.xlabel('Start of Season Price'); plt.ylabel('Year influence')
plt.savefig('static/startvsyear')
plt.show()

def getprice(value,minval,inflation):
    l1=[0,0,1,value,minval,inflation]
    print ("Year the item value belongs to : ",int(regression1.predict(np.reshape(l1, (-1,len(l1))))))
getprice(890,880,8.3)


from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
sc_y=StandardScaler()

from sklearn.svm import SVR
svr_regressor = SVR(kernel='rbf')
svr_regressor.fit(sc_x.fit_transform(x_tr), y_tr)

plt.scatter(x_tr[:,4], y_tr[:,-1], color='grey')
plt.plot(x_tr, svr_regressor.predict(x_tr), color = 'blue' )
plt.xlabel('Item prices');plt.ylabel('Year');plt.title('Scaled SVR value')
plt.savefig('static/itemvsyear')
plt.show()
def getsvrprice(value,minval,inflation):
    l2=[0,0,1,value,minval,inflation]
    print ("Year the item value belongs to (SVR): ",int(svr_regressor.predict(np.reshape(l2, (-1,len(l2))))))
getsvrprice(890,880,8.3)

def geterror(mlreg,svreg):
    error_perc,er_p=0,0
    for i in range(len(mlreg)):
        error_perc+= abs(Y[i]-mlreg[i]) / max(mlreg[i],Y[i])

    print ("Standard error field for MLR : ",error_perc)
    for j in range(len(svreg)):
        er_p+= abs(Y[i] - svreg[i]) / max(Y[i],svreg[i])

    print ("Standard error field for SVR : ",er_p)
    return [er_p,error_perc]
a=geterror(y_predict, svr_regressor.predict(x_tr))

from sklearn.externals import joblib
joblib.dump(svr_regressor,'predmodel.pkl')
joblib.dump(regression1,'premodel1.pkl')
