
#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score
import statsmodels.api as sm

# veri yukleme
veriler = pd.read_csv('maaslar_yeni.csv')

x=veriler.iloc[:,2:5]
y=veriler.iloc[:,5:]
X=x.values #x içindeki verileri numpy dizisine çevir
Y=y.values

print(veriler.corr())#korelasyon için 

#lineer regresyon
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)# Xi Ye öğret modeli


model=sm.OLS(lin_reg.predict(X),X)#x ile tahmin arasındaki ilişkiyi kontrol
print(model.fit().summary())#modelin istatistiksel sonuclarını yazdırır 

print("Lineer R2 Değeri")
print(r2_score(Y,lin_reg.predict(X)))

#polinomial regresyon 
from sklearn.preprocessing import PolynomialFeatures 
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(X)#polinomal şekle dönüştürmek
print(x_poly)

lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#tahmin
print("Poly OLS")
model2=sm.OLS(lin_reg.predict(poly_reg.fit_transform(X)),X)
print(model2.fit().summary())

print("Polinominal R2 değeri")
print(r2_score(Y,lin_reg2.predict(poly_reg.fit_transform(X))))



#veri olceklernmesi
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=np.ravel(sc2.fit_transform(Y.reshape(-1,1)))#önce standart sonra tek boyut hale 


from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

print("SVR OLS")
model3=sm.OLS(svr_reg.predict(x_olcekli),x_olcekli)
print(model.fit().summary())

print("SVR 2 DEğeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

#Karar Ağacı Yapısı

from sklearn.tree import DecisionTreeRegressor
r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

print("Karar Ağacı Çiz")
model4=sm.OLS(r_dt.predict(X),X)
print(model4.fit().summary())

print("Karar Ağacı 2 değeri")
print(r2_score(Y,r_dt.predict(X)))

#Rassal Orman Regresyon

from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0)#kaç ağaç ve başlangıç değeri
rf_reg.fit(X,Y.ravel())

print("Random Forest OLS")
model5=sm.OLS(rf_reg.predict(X),X)
print(model.fit().summary)

print("Random Forest R2 Değeri")
print(r2_score(Y,rf_reg.predict(X)))

#Hepsinin Sonuclari
print('-----------------------')
print('Linear R2 degeri')
print(r2_score(Y, lin_reg.predict(X)))

print('Polynomial R2 degeri')
print(r2_score(Y, lin_reg2.predict(poly_reg.fit_transform(X))))

print('SVR R2 degeri')
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)))


print('Decision Tree R2 degeri')
print(r2_score(Y, r_dt.predict(X)))

print('Random Forest R2 degeri')
print(r2_score(Y, rf_reg.predict(X)))











