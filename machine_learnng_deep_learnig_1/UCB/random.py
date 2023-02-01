#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:03:45 2018

@author: sadievrenseker
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Ads_CTR_Optimisation.csv')

import random

N = 10000
d = 10 
toplam = 0
secilenler = []
for n in range(0,N): #kaç kere tıklandıgı
    ad = random.randrange(d) #kaç tanesine tıklandığı ilan sayısı
    odul=veriler.values[n,ad]  #verilerdeki n. satır=1 ise ödül 1
    toplam=toplam+odul 
    
plt.hist(secilenler)
plt.show()      
    










