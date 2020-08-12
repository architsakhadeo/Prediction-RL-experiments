import numpy as np
import scipy.stats as st

# Returns the mean of the runs per timestep
def getMean(data):
    return np.mean(data, axis=0)

# Returns the standard deviation of the runs per timestep
def getStddev(data):
    return np.std(data, axis=0)

# Returns the median run
def getMedian(data):
    numRuns = len(data)

    # Sorts runs w.r.t. final performance in an ascending manner
    indices = np.argsort(data[:,-1])

    if numRuns % 2 == 1:
        # Odd number of runs, select middle run
        median = data[indices[int((numRuns-1)/2)]]
    else:
        # Even number of runs, select average of the two runs in the middle
        median = (data[indices[int(numRuns/2)]] + data[indices[int(numRuns/2 - 1)]])/2.0
    return median

# Returns the run with the best performance (highest return or least failures)
def getBest(data, transformation='Returns'):
    numRuns = len(data)

    # Sorts runs w.r.t. final performance in an ascending manner
    indices = np.argsort(data[:,-1])

    # Change this code carefully
    if transformation == 'Failures':
        best = data[indices[0]]
    else:
        best = data[indices[-1]]
    return best

# Returns the run with the worst performance (lowest return or most failures)
def getWorst(data, transformation='Returns'):
    numRuns = len(data)

    # Sorts runs w.r.t. final performance in an ascending manner
    indices = np.argsort(data[:,-1])

    # Change this code carefully
    if transformation == 'Failures':
        worst = data[indices[-1]]
    else:
        worst = data[indices[0]]
    return worst

# Returns the confidence interval of the mean
# Make this code adaptive using pandas/scipy so it finds the critical value given the confidence
# confidence should be strictly less than 1.0
def getConfidenceIntervalOfMean(data, confidence=0.95):
    numRuns = len(data)
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    criticalValue = st.norm.ppf( (confidence + 1.0) / 2.0 )
    confidenceInterval = criticalValue * stddev / np.sqrt(numRuns)
    lowerbound = mean - confidenceInterval
    upperbound = mean + confidenceInterval
    return lowerbound, upperbound

# Sorts the performances in an ascending manner
# Returns the lower and upper performances based on their percentile
def getRegion(data, lower=0.0, upper=1.0):
    numRuns = len(data)

    # Sorts runs w.r.t. final performance in an ascending manner
    indices = np.argsort(data[:,-1])
    lowerIndex = int(round( lower * ( numRuns-1 ) ))
    upperIndex = int(round( upper * ( numRuns-1 ) ))

    '''
    # Change this code carefully
    if transformation == 'Failures':
        # li and ui are important and not redundant. They act like temp variables
        li = lowerIndex
        ui = upperIndex
        lowerIndex = numRuns - 1 - ui
        upperIndex = numRuns - 1 - li
    '''
        
    lowerRun = data[indices[lowerIndex]]
    upperRun = data[indices[upperIndex]]
    return lowerRun, upperRun