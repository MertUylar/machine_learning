# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#kodlar
#veri yukleme

veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")

print(veriler)

#veri on isleme
aylar= veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)
#satilar bağımlı aylar bağımsız değişken

satislar2 = veriler.iloc[:,0:1].values
print(satislar)



from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(aylar,satislar,test_size=0.33, random_state=0)

# verileri aynı dünyadan yapıp standart sapmalarını azaltmak amac
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

y_train=sc.fit_transform(y_train)
y_test=sc.fit_transform(y_test)


from sklearn.linear_model import LinearRegression
#x ve y train bilgilerini alarak bir model nsa et

#model insşası(linear regration)
lr=LinearRegression()
lr.fit(x_train,y_train)

























