import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer 
from sklearn import preprocessing


veriler=pd.read_csv('veriler.csv')
ulke=veriler.iloc[:,0:1].values
print(ulke)


#fit ve transform formlarını cagırmak için bunu yaptık aynanda kullanabil.
le=preprocessing.LabelEncoder()
#colondaki her şeyi almak için boş bırakıyoruz.
#önce 0 1 2 ye çevirdi
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])

print(ulke)
#geçerli olanları 1 geçersizleri 0 a döndrdü
ohe=preprocessing.OneHotEncoder()

ulke=ohe.fit_transform(ulke).toarray()

print(ulke)




