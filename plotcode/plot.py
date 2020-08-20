#!/usr/bin/env python
# coding: utf-8

# # Plot learning curve

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from stats import getMean, getMedian, getBest, getWorst, getConfidenceIntervalOfMean, getRegion

# Add color, linestyles as needed

def plotMean(xAxis, data, color, label):
    mean = getMean(data)
    plt.plot(xAxis, mean, color=color, label=label)

def plotMedian(xAxis, data, color):
    median = getMedian(data)
    plt.plot(xAxis, median, label=alg+'-median', color=color)

def plotBest(xAxis, data, transformation, color):
    best = getBest(data, transformation)
    plt.plot(xAxis, best, label=alg+'-best', color=color)

def plotWorst(xAxis, data, transformation, color):
    worst = getWorst(data,  transformation)
    plt.plot(xAxis, worst, label=alg+'-worst', color=color)

def plotMeanAndConfidenceInterval(xAxis, data, confidence, color, label):
    plotMean(xAxis, data, color=color, label=label)
    lowerBound, upperBound = getConfidenceIntervalOfMean(data, confidence)
    plt.fill_between(xAxis, lowerBound, upperBound, alpha=0.25, color=color)

def plotMeanAndPercentileRegions(xAxis, data, lower, upper, color, label):
    plotMean(xAxis, data, color, label)
    lowerRun, upperRun = getRegion(data, lower, upper)
    plt.fill_between(xAxis, lowerRun, upperRun, alpha=0.25, color=color, label=label)

labels = ['trunc=1', 'trunc=2', 'trunc=4']#, 'trunc=6', 'trunc=10', 'trunc=16']
paths = ['../Data/RingWorldSweepData/ringsize=6_gamma=0.0_hiddenunits=6_' + labels[i] + '_learnrate=0.0005/' for i in range(len(labels))]

type = 'Simple' # Simple or Random

for path in paths:
    files = os.listdir(path)
    lossesPickled = [file for file in files if type in file and 'loss' in file]
    observationsPickled = [file for file in files if type in file and 'observations' in file]

    losses = []
    observations = []

    for i in range(len(lossesPickled)):
        losses.append(pickle.load(open(path+lossesPickled[i], 'rb')))
        observations.append(pickle.load(open(path+observationsPickled[i], 'rb')))

    print(losses)


    losses0 = []
    losses1 = []

    for i in range(len(losses)):
        temp_losses0 = []
        temp_losses1 = []
        
        for j in range(len(losses[i])):
            
            if observations[i][j] == 0:
                temp_losses0.append(losses[i][j])
                temp_losses1.append(-1)
                
            elif observations[i][j] == 1:
                temp_losses1.append(losses[i][j])
                temp_losses0.append(-1)

        losses0.append(temp_losses0)
        losses1.append(temp_losses1)



    #Binning - bin size = 100

    bin_size = 1000

    binned_losses0 = []
    binned_losses1 = []
    binned_losses = []


    for i in range(len(losses0)):
        temp_binned_losses0 = []
        
        for j in range(0, len(losses0[i]), bin_size):
            current = j
            next = j + bin_size
            
            if next > len(losses0[i]):
                next = len(losses0[i])
                
            sum = 0
            count = 0
            for k in range(current, next):
                if losses0[i][k] != -1:
                    sum += losses0[i][k]
                    count += 1
            
            if count == 0:
                average = 0
            else:
                average = sum * 1.0 / count
            
            for k in range(current, next):
                temp_binned_losses0.append(average)
                
        binned_losses0.append(temp_binned_losses0)

    for i in range(len(losses1)):
        temp_binned_losses1 = []
        
        for j in range(0, len(losses1[i]), bin_size):
            current = j
            next = j + bin_size
            if next > len(losses1[i]):
                next = len(losses1[i])
                
            sum = 0
            count = 0
            for k in range(current, next):
                if losses1[i][k] != -1:
                    sum += losses1[i][k]
                    count += 1
                    
            if count == 0:
                average = 0
            else:
                average = sum * 1.0 / count

            for k in range(current, next):
                temp_binned_losses1.append(average)
                
        binned_losses1.append(temp_binned_losses1)


    sliding_window = 1000
    losses = np.array(losses)
    binned_losses0 = np.array(binned_losses0)
    binned_losses1 = np.array(binned_losses1)

    import numpy as np

    sliding_losses = []
    sliding_losses0 = []
    sliding_losses1 = []

    for i in range(len(losses)):
        sliding_losses.append(list(np.convolve(losses[i], np.ones(sliding_window)/sliding_window, 'valid')))

    for i in range(len(binned_losses0)):
        sliding_losses0.append(list(np.convolve(binned_losses0[i], np.ones(sliding_window)/sliding_window, 'valid')))

    for i in range(len(binned_losses1)):
        sliding_losses1.append(list(np.convolve(binned_losses1[i], np.ones(sliding_window)/sliding_window, 'valid')))    


    plot_losses = np.array([np.array(i) for i in sliding_losses])
    plot_losses0 = np.array([np.array(i) for i in sliding_losses0])
    plot_losses1 = np.array([np.array(i) for i in sliding_losses1])
    timesteps_individual = np.array([i+1 for i in range(len(plot_losses0[0]))])
    timesteps_both = np.array([i+1 for i in range(len(plot_losses[0]))])



    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


    #plotMeanAndConfidenceInterval(timesteps_individual + sliding_window - 1, plot_losses1, 0.95, color=colors[2], label='1 observation mean')
    #plotMeanAndConfidenceInterval(timesteps_individual + sliding_window - 1, plot_losses0, 0.95, color=colors[3], label='0 observation mean')
    plotMeanAndConfidenceInterval(timesteps_both + sliding_window - 1, plot_losses, 0.95, color=colors[paths.index(path)], label=labels[paths.index(path)])

plt.title('Ringsize=6, 30 runs', pad=25, fontsize=10)
plt.xlabel('Timesteps', labelpad=35)
plt.ylabel('Prediction error \n (RMSVE)', rotation=0, labelpad=45)
plt.rcParams['figure.figsize'] = [8, 5.33]
plt.legend(loc=0)
plt.yticks()
plt.xticks()
plt.tight_layout()
plt.show()
#plt.savefig('../images/test5.png',dpi=500, bbox_inches='tight')

