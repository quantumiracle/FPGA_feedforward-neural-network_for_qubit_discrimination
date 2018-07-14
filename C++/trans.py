import numpy as np
import random
import cPickle as pickle
import matplotlib.pyplot as plt
import argparse
import math
import gzip

f =gzip.open('./DetectionBinsData_pickle615_clean.gzip','rb')  #49664*100 times measurement
fb=open('bright.txt','w')
fd=open('dark.txt','w')
num_bins=100

for k in range(10000):
    print(k)
    d_data=pickle.load(f)[:,1:num_bins+1]
    b_data=pickle.load(f)[:,102:102+num_bins]
    #print(d_data,b_data)

    for i in range(100):
        for j in range(num_bins):
            fd.write(str(d_data[i][j]))
            fb.write(str(b_data[i][j]))
        fd.write('\n')
        fb.write('\n')

fd.close()
fb.close()