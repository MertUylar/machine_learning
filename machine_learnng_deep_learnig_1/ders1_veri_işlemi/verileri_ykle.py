import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv('veriler.csv')
print(veriler)
#veri ön işleme
boy= veriler[["boy"]]
print(boy)


bk= veriler[["boy","kilo"]]
print(bk)



