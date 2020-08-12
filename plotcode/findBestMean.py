import pickle
import os
import numpy as np

dirpaths = ['../Data/RingWorldData/trunc1/', '../Data/RingWorldData/trunc2/', '../Data/RingWorldData/trunc3/', 
'../Data/RingWorldData/trunc4/', '../Data/RingWorldData/trunc5/', '../Data/RingWorldData/trunc6/', 
'../Data/RingWorldData/trunc7/', '../Data/RingWorldData/trunc8/', '../Data/RingWorldData/trunc9/', '../Data/RingWorldData/trunc10/']

for dirpath in dirpaths:
    subdirs = os.listdir(dirpath)
    minLoss = 100
    bestparam = 'None'

    for i in range(len(subdirs)):
        subdirpath = dirpath + subdirs[i] + '/'
        files = os.listdir(subdirpath)
        avgFileLoss = 0
        count = 0
        for j in range(len(files)):
            if 'observations' in files[j]:
                continue
            
            avgFileLoss += np.mean(pickle.load(open(subdirpath+files[j],'rb'))[-1000:])
            count += 1
        avgFileLoss /= count
        if avgFileLoss < minLoss:
            minLoss = avgFileLoss
            bestparam = subdirpath

    #print(minLoss)
    print(bestparam)


