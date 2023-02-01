import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('eksikveriler.csv')
print(veriler)
#veri ön işleme
boy= veriler[["boy"]]
print(boy)


bk= veriler[["boy","kilo"]]
print(bk)

#eksik veriler

from sklearn.impute import SimpleImputer 
#nan olanları meanla değiştir yanı ortalamayla değiştir
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
#1. ve 4. kolonlar geliyor
Yas=veriler.iloc[:,1:4].values
#yaşın 1 den 4 e kadar olan kolonlarını öğren ortalama değer yani
imputer=imputer.fit(Yas[:,1:4])

#non olan kolonlar fit sayesinde öğrenlenlerle değiştirilecek
Yas[:,1:4]=imputer.transform(Yas[:,1:4])

print(Yas)

