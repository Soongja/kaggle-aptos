# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Any results you write to the current directory are saved as output.

import os
import pandas as pd
from io import StringIO

s = """id_code,diagnosis

"""

s = StringIO(s)
predict = pd.read_csv(s)

try:
    test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
except:
    test_df = pd.read_csv("../input/test.csv")

sub = pd.merge(test_df, predict, on='id_code', how='left').fillna(0)
sub["diagnosis"] = sub["diagnosis"].astype(int)
sub.to_csv("submission.csv", index=False)