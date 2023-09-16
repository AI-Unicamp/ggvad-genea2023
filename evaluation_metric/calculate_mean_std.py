'''
calculate mean and std of the train dataset
'''

import glob
import numpy as np

files = glob.glob(f'../dataset/genea2023_dataset/trn/main-agent/npy/*.npy')
files = files[::5]  # use a subset to avoid out of memory

all_data = []
for file in files:
    data = np.load(file)
    all_data.append(data)

all_data = np.vstack(all_data)

print(all_data.shape)

mean = np.mean(all_data, axis=0)
std = np.std(all_data, axis=0)
print(mean.shape)
print(std.shape)

print(*mean, sep=',')
print(*std, sep=',')
