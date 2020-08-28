import pickle
import os
import numpy as np

basepath = '../Data/RingWorldSweepData/ringsize=6/'
dirpaths = os.listdir(basepath)

for dirpath in dirpaths:
    subdirs = os.listdir(basepath+dirpath)
    minLoss = 100
    bestparam = 'None'

    for i in range(len(subdirs)):
        subdirpath = basepath+dirpath+'/'+subdirs[i] + '/'
        extrapath = subdirpath + 'RingWorldSweepData/'+subdirs[i]+'/'
        files = os.listdir(extrapath)
        avgFileLoss = 0
        count = 0
        for j in range(len(files)):
            if 'observations' in files[j]:
                continue
            
            avgFileLoss += np.mean(pickle.load(open(extrapath+files[j],'rb'))[-50000:])
            count += 1
        avgFileLoss /= count
        if avgFileLoss < minLoss:
            minLoss = avgFileLoss
            bestparam = subdirpath

    print(bestparam, minLoss)


