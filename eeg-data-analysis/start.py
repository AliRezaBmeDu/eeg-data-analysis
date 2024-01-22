# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 00:54:39 2024

@author: Mohammad Ali Reza
"""

import os
import pandas as pd, numpy as np
from glob import glob
import matplotlib.pyplot as plt

BASE_PATH = '../hms-harmful-brain-activity-classification/'

df = pd.DataFrame({'path': glob(BASE_PATH + '**/*.parquet')})
df['test_type'] = df['path'].str.split('/').str.get(-2).str.split('_').str.get(-1)
df['id'] = df['path'].str.split('/').str.get(-1).str.split('.').str.get(0)

df_eeg = pd.read_parquet(BASE_PATH + 'train_eegs/1000913311.parquet')
df_eeg.head()


# Determine the number of channels
# Assuming each row is a time point and each column is a channel
n_channels = df_eeg.shape[1]
n_channels

df = pd.read_csv(BASE_PATH + 'train.csv')
TARGETS = df.columns[-6:]
print('Train shape:', df.shape )
print('Targets', list(TARGETS))
df.head()


