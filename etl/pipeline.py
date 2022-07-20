
import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

import os
os.chdir("C:\Work\Data Science\Openclassrooms\projet 7\work\\etl")

from etl.functions import application_train_test, bureau_and_balance, previous_applications, installments_payments, pos_cash, credit_card_balance

num_rows = None


#df = application_train_test(num_rows)
df = application_train_test(num_rows,nan_as_category=True)



with timer("Process bureau and bureau_balance"):
    bureau = bureau_and_balance(num_rows)
    print("Bureau df shape:", bureau.shape)
    #df = df.join(bureau, how='left', on='SK_ID_CURR')
    df = df.merge(bureau, how='left', on='SK_ID_CURR', indicator="merge_with_bureau")
    del bureau
    gc.collect()


with timer("Process previous_applications"):
    prev = previous_applications(num_rows)
    print("Previous applications df shape:", prev.shape)
    #df = df.join(prev, how='left', on='SK_ID_CURR')
    df = df.merge(prev, how='left', on='SK_ID_CURR', indicator="merge_with_prev")
    del prev
    gc.collect()


with timer("Process POS-CASH balance"):
    pos = pos_cash(num_rows)
    print("Pos-cash balance df shape:", pos.shape)
    #df = df.join(pos, how='left', on='SK_ID_CURR')
    df = df.merge(pos, how='left', on='SK_ID_CURR', indicator="merge_with_pos")
    del pos
    gc.collect()


with timer("Process installments payments"):
    ins = installments_payments(num_rows)
    print("Installments payments df shape:", ins.shape)
    #df = df.join(ins, how='left', on='SK_ID_CURR')
    df = df.merge(ins, how='left', on='SK_ID_CURR', indicator="merge_with_ins")
    del ins
    gc.collect()


with timer("Process credit card balance"):
    cc = credit_card_balance(num_rows)
    print("Credit card balance df shape:", cc.shape)
    #df = df.join(cc, how='left', on='SK_ID_CURR')
    df = df.merge(cc, how='left', on='SK_ID_CURR', indicator="merge_with_cc")
    del cc
    gc.collect()

with open('..\\data\\output\\loans.csv', 'w', encoding='UTF8', newline='') as f:
    df.to_csv(f, index=False)



