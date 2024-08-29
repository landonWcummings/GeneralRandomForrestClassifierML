import numpy as np
import pandas as pd

from backend.RFCplace import RFC
ts = 4
if ts ==1:
    trainpath = 'C:/Users/lndnc/Downloads/titanic/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/titanic/test.csv'
    target = 'Survived'
    savehere = 'C:/Users/lndnc/Downloads/titanic/submissiongeneral.csv'

if ts==2:
    trainpath = 'C:/Users/lndnc/Downloads/playground-series-s4e8/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/playground-series-s4e8/test.csv'
    target = 'class'
    savehere = 'C:/Us ers/lndnc/Downloads/playground-series-s4e8/submissiongeneral.csv'



if ts ==3:
    trainpath = 'C:/Users/lndnc/Downloads/spaceship-titanic/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/spaceship-titanic/test.csv'
    target = 'Transported'
    savehere = 'C:/Users/lndnc/Downloads/spaceship-titanic/submissiongeneral.csv'

if ts == 4:
    #C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques
    trainpath = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/test.csv'
    target = 'SalePrice'
    savehere = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/submissiongeneral1110000.csv'



randomforrestclassifier = RFC(trainpath, predictpath, 
                              target, savehere, num_estimators=[5000],
                              depths=[None],min_samples_split=[5],split=0.02,cramers_v_cut=0.1)
randomforrestclassifier.complete()
