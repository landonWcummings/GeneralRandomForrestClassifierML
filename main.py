import numpy as np
import pandas as pd

from backend.RFCplace import RFC
ts = 5
breakup = []
if ts ==1:
    trainpath = 'C:/Users/lndnc/Downloads/titanic/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/titanic/test.csv'
    target = 'Survived'
    savehere = 'C:/Users/lndnc/Downloads/titanic/submissiongeneral.csv'

if ts==2:
    trainpath = 'C:/Users/lndnc/Downloads/playground-series-s4e8/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/playground-series-s4e8/test.csv'
    target = 'class'
    savehere = 'C:/Users/lndnc/Downloads/playground-series-s4e8/submissiongeneral.csv'

if ts ==3:
    trainpath = 'C:/Users/lndnc/Downloads/spaceship-titanic/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/spaceship-titanic/test.csv'
    target = 'Transported'
    savehere = 'C:/Users/lndnc/Downloads/spaceship-titanic/submissiongeneral.csv'
    breakup = ["Cabin",'/']

if ts == 4:
    trainpath = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/test.csv'
    target = 'SalePrice'
    savehere = 'C:/Users/lndnc/Downloads/house-prices-advanced-regression-techniques/submissiongeneral.csv'

if ts == 5:
    trainpath = 'C:/Users/lndnc/Downloads/kagglecardatacomp/train.csv'
    predictpath = 'C:/Users/lndnc/Downloads/kagglecardatacomp/test.csv'
    target = 'price'
    savehere = 'C:/Users/lndnc/Downloads/kagglecardatacomp/submissiongeneral.csv'
    breakup = ["engine", "HP"]

mod = 1
if mod == 1:
    randomforrestclassifier = RFC(trainpath, predictpath, 
                                target, savehere, num_estimators=[300],
                                depths=[None],min_samples_split=[3],split=0.02,
                                cramers_v_cut=0.21, breakup=breakup,require_individual_correlation=True,
                                exclude_values_limit=3000, include_nan=True)
    randomforrestclassifier.complete()

