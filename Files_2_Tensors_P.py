from multiprocessing import Pool
import os
import numpy as np
from glob import glob

################ MODIFY HERE ###################
data_path = "DATA PATH HERE"
train_frac = 0.95
len_data = 1001 #Length of data
# len_data = 401 #Length of data
len_label = 14 #Length of label #10 sp + dw + t + fwhm + dynamic range
################################################

files = glob(data_path+"data.*.txt")
try:
    os.mkdir(data_path + "ndarray")
except:
    pass
nbrs = [file.split(".")[-2] for file in files]

Ndata = len(nbrs)
data = np.zeros((int(Ndata), len_data))
label = np.zeros((int(Ndata), len_label))

def read_from_disk(nb):
    data_file = np.loadtxt(data_path + "data." + nb + ".txt")
    label_file = np.loadtxt(data_path + "label." + nb + ".txt")

    return data_file, label_file

pool = Pool()
for i, out in enumerate(pool.imap(read_from_disk, nbrs)):
    print(i)
    data[i,:] = out[0]
    label[i,:] = out[1]
pool.close()
pool.join()

np.savez(data_path+'ndarray/train', train_data=data[:int(Ndata*train_frac)], train_label=label[:int(Ndata*train_frac)])
np.savez(data_path+'ndarray/test', test_data=data[int(Ndata*train_frac):], test_label=label[int(Ndata*train_frac):])

print("Done")
