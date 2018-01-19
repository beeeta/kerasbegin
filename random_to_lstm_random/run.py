
import random
from collections import Counter
from keras.utils import np_utils
import numpy as np


big_numb = 100

row_data = [random.randint(1,big_numb) for _ in range(1000*1000)]

# 先验证原始数据的随机性

count = Counter(row_data)
with open('row_random_static.txt','w') as f:
    for (i,j) in dict(count).items():
        str_print = '{}:{}%'.format(i,j*100/len(row_data))
        print(str_print)
        f.write(str_print+'\n')

row_len = len(row_data)
seq_len = 100

xdata = []
ydata = []
for i in range(0,row_len-seq_len,1):
    xdata.append([row_data[i:i+seq_len]])
    ydata.append(row_data[i+seq_len])

patterns = len(xdata)
X = xdata/big_numb
X = np.reshape(X,(patterns,seq_len,1))
Y = np_utils.to_categorical(ydata)


