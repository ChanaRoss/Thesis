import numpy as np
from matplotlib import  pyplot as plt
from sklearn import metrics
from math import sqrt
import seaborn as sns
sns.set()

def createEventDistributionUber(simStartTime, startTime, endTime, probabilityMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = probabilityMatrix[x, y, t + firstTime, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                #for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos  = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:'+str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow

def createRealEventsDistributionUber(simStartTime, startTime, endTime, eventsMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(eventsMatrix.shape[0]):
            for y in range(eventsMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                numEvents = eventsMatrix[x, y, t + firstTime]
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                # for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:' + str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow

def getBenchmarkAccuracy(start_time, seq_len, realMat, probMat):
    realMatOut = realMat[:, :, start_time + seq_len+1]
    distMatOut = np.zeros_like(realMatOut)
    for x in range(probMat.shape[0]):
        for y in range(probMat.shape[1]):
            randNum      = np.random.uniform(0, 1)
            cdfNumEvents = probMat[x, y, start_time + seq_len + 1, :]
            # find how many events are happening at the same time
            numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
            numEvents = np.floor(numEvents).astype(int)
            distMatOut[x, y] = numEvents
    return realMatOut, distMatOut



def main():
    # data loader -
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileNameReal = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_allData.p'
    fileNameDist = '4D_UpdatedGrid_5min_250Grid_LimitedProbability_CDF_mat_allData_Benchmark.p'
    # data real values are between 0 and k (k is the maximum amount of concurrent events at each x,y,t)
    # data dist have values that are the probability of having k events at x,y,t
    dataInputReal = np.load(path + fileNameReal)  # matrix size is : [xsize , ysize, timeseq]
    dataInputProb = np.load(path + fileNameDist)  # matrix size is : [xsize , ysize, timeseq, probability for k events]
    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model
    dataInputProb = dataInputProb[xmin:xmax, ymin:ymax, :, :]
    accuracy = []
    accuracy1 = []
    rmse = []
    for i in range(300):
        start_time = np.random.randint(10, dataInputProb.shape[2]-10)
        realMatOut, distMatOut = getBenchmarkAccuracy(start_time, 5, dataInputReal, dataInputProb)
        rmse.append(sqrt(metrics.mean_squared_error(realMatOut.reshape(-1), distMatOut.reshape(-1))))
        accuracy.append(np.sum(realMatOut == distMatOut)/(realMatOut.shape[0]*realMatOut.shape[1]))
        realMatOut[realMatOut > 1] = 1
        distMatOut[distMatOut > 1] = 1
        accuracy1.append(np.sum(np.sum(realMatOut == distMatOut)/(realMatOut.shape[0]*realMatOut.shape[1])))
    plt.scatter(range(len(accuracy)), 100*np.array(accuracy))
    plt.xlabel('run number [#]')
    plt.ylabel('accuracy [%]')
    plt.figure()
    plt.scatter(range(len(rmse)), np.array(rmse))
    plt.xlabel('run number [#]')
    plt.ylabel('RMSE')
    plt.figure()
    plt.scatter(range(len(accuracy1)), 100*np.array(accuracy1))
    print("average RMSE for 300 runs is:"+str(np.mean(np.array(rmse))))
    print("average accuracy for 300 runs is:"+str(np.mean(np.array(accuracy))))
    print("average corrected accuracy for 300 runs is:" + str(np.mean(np.array(accuracy1))))

    plt.show()
    return




if __name__ == '__main__':
    main()
    print('Done.')