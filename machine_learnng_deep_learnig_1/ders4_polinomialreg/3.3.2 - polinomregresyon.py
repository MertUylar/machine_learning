# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# veri yukleme
veriler = pd.read_csv('maaslar.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X,Y,color="red") #2 boyulu uzaya dağıtma
plt.plot(x,lin_reg.predict(x),color="blue")# x e karşılık gelen tahminnleri görselleştir
plt.show()

from sklearn.preprocessing import PolynomialFeatures# sayıyı polinom yapar
poly_reg=PolynomialFeatures(degree=2)
X_poly =poly_reg.fit_transform(X)

print(X_poly) 

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)
plt.scatter(X,Y,color="red")
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color="blue") #polinom dönüşünü yap ve çiz
plt.show()
