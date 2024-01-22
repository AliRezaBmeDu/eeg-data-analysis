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


