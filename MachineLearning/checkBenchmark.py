import numpy as np
from matplotlib import  pyplot as plt
from sklearn import metrics
from math import sqrt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 11)})

import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif


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
    _, probStartTime = np.divmod(start_time + seq_len+1, probMat.shape[2])
    realMatOut = realMat[:, :, 48 + start_time + seq_len+1]
    distMatOut = np.zeros_like(realMatOut)
    for x in range(probMat.shape[0]):
        for y in range(probMat.shape[1]):
            randNum      = np.random.uniform(0, 1)
            cdfNumEvents = probMat[x, y, probStartTime, :]
            # find how many events are happening at the same time
            numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
            numEvents = np.floor(numEvents).astype(int)
            distMatOut[x, y] = numEvents
    return realMatOut, distMatOut


def plotSpesificTime(dataReal, dataPred, t, fileName):
    day         = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week        = np.floor_divide(t, 2 * 24 * 7) + 14 # first week is week 14 of the year
    hour, temp  = np.divmod(t, 2)
    hour        += 8  # data starts from 98 but was normalized to 0
    _, hour        = np.divmod(hour, 24)
    minute      = temp * 30

    dataFixed   = np.zeros_like(dataReal)
    dataFixed   = np.swapaxes(dataReal, 1, 0)
    dataFixed   = np.flipud(dataFixed)

    dataFixedPred = np.zeros_like(dataPred)
    dataFixedPred = np.swapaxes(dataPred, 1, 0)
    dataFixedPred = np.flipud(dataFixedPred)

    f, axes = plt.subplots(1, 2)
    ticksDict = list(range(21))

    sns.heatmap(dataFixed, cbar = False, center = 1, square=True, vmin = 0, vmax = 20, ax=axes[0], cmap = 'CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    sns.heatmap(dataFixedPred, cbar=True, center=1, square=True, vmin=0, vmax=20, ax=axes[1], cmap='CMRmap_r',cbar_kws=dict(ticks=ticksDict))
    axes[0].set_title('week- {0}, day- {1},time- {2}:{3}'.format(week,day,hour, minute) + ' , Real data')
    axes[1].set_title('Predicted data')
    # plt.title('time is -  {0}:{1}'.format(hour, minute))
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel('Y axis')
    axes[1].set_xlabel('X axis')
    axes[1].set_ylabel('Y axis')
    plt.savefig(fileName + '_' + str(t) +'.png')
    plt.close()
    return

def main():
    # data loader -
    path = '/Users/chanaross/dev/Thesis/UberData/'
    figPath = '/Users/chanaross/dev/Thesis/MachineLearning/BenchmarkFigures/'
    fileNameReal = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'
    # fileNameReal = '4D_matPerWeek_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'
    fileNameDist = '4D_ProbabilityMat_allDataLatLonCorrected_20MultiClass_CDF_500gridpickle_30min.p'
    # data real values are between 0 and k (k is the maximum amount of concurrent events at each x,y,t)
    # data dist have values that are the probability of having k events at x,y,t
    dataInputReal = np.load(path + fileNameReal)  # matrix size is : [xsize , ysize, timeseq]
    dataInputProb = np.load(path + fileNameDist)  # matrix size is : [xsize , ysize, timeseq, probability for k events]

    # dataInputRealFull = np.zeros(shape=(dataInputReal.shape[0], dataInputReal.shape[1], dataInputReal.shape[2] * dataInputReal.shape[3]))
    # numTimes = dataInputReal.shape[2]
    # for i in range(dataInputReal.shape[3]):
    #     dataInputRealFull[:, :, i*numTimes: (i+1)*numTimes] = dataInputReal[:, :, :, i]
    # dataInputRealFull.dump('/Users/chanaross/dev/Thesis/UberData/3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p')

    xmin = 0
    xmax = dataInputReal.shape[0]
    ymin = 0
    ymax = dataInputReal.shape[1]
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model
    dataInputProb = dataInputProb[xmin:xmax, ymin:ymax, :, :]
    accuracy = []
    accuracy1 = []
    rmse = []
    numEventsPredicted = []
    numEventsCreated = []

    correct_zeros = []
    correct_non_zeros = []

    timeOut = []
    numRuns = 10
    fileNameOut = '500grid_30min_20Multi_benchmark_results_random_' + str(numRuns)
    for i in range(numRuns):
        start_time = np.random.randint(10, dataInputReal.shape[2]-10)
        # start_time = 11
        timeOut.append(start_time)
        realMatOut, distMatOut = getBenchmarkAccuracy(start_time, 5, dataInputReal, dataInputProb)
        sizeMat = realMatOut.shape[0] * realMatOut.shape[1]
        rmse.append(sqrt(metrics.mean_squared_error(realMatOut.reshape(-1), distMatOut.reshape(-1))))
        accuracy.append(np.sum(realMatOut == distMatOut)/(sizeMat))
        sizeMat_zeros = distMatOut[realMatOut == 0].size
        sizeMat_non_zeros = distMatOut[realMatOut != 0].size
        if (sizeMat_non_zeros > 0):
            correct_non_zeros.append(np.sum(distMatOut[realMatOut != 0] == realMatOut[realMatOut != 0]) / sizeMat_non_zeros)
        if sizeMat_zeros > 0:
            correct_zeros.append(np.sum(distMatOut[realMatOut == 0] == realMatOut[realMatOut == 0]) / sizeMat_zeros)
        plotSpesificTime(realMatOut, distMatOut, start_time, figPath + fileNameOut)
        numEventsCreated.append(np.sum(realMatOut))
        numEventsPredicted.append(np.sum(distMatOut))

        # realMatOut[realMatOut > 1] = 1
        # distMatOut[distMatOut > 1] = 1
        # accuracy1.append(np.sum(np.sum(realMatOut == distMatOut)/(realMatOut.shape[0]*realMatOut.shape[1])))
        # if (realMatOut[realMatOut!=0].size >0):
        #     non_zero_accuracy1.append(np.sum(np.sum(realMatOut[realMatOut != 0] == distMatOut[realMatOut != 0]))/(realMatOut[realMatOut != 0].size))
        #
        # if (distMatOut[distMatOut!=0].size >0):
        #     non_zero_accuracy1_dist.append(np.sum(np.sum(realMatOut[distMatOut != 0] == distMatOut[distMatOut != 0]))/(realMatOut[distMatOut != 0].size))


    listNames = [fileNameOut + '_' + str(t) + '.png' for t in timeOut]
    create_gif(figPath, listNames, 1, fileNameOut)

    plt.scatter(range(len(accuracy)), 100*np.array(accuracy))
    plt.xlabel('run number [#]')
    plt.ylabel('accuracy [%]')
    plt.figure()
    plt.scatter(range(len(rmse)), np.array(rmse))
    plt.xlabel('run number [#]')
    plt.ylabel('RMSE')
    plt.figure()
    plt.scatter(range(len(numEventsCreated)), np.array(numEventsCreated), label="num real events")
    plt.scatter(range(len(numEventsPredicted)), np.array(numEventsPredicted), label="num predicted")
    plt.xlabel('run number [#]')
    plt.ylabel('num events created')
    plt.legend()
    plt.figure()
    plt.scatter(range(len(numEventsCreated)), np.abs(np.array(numEventsCreated) - np.array(numEventsPredicted)), label="difference between prediction and real")
    plt.xlabel('run number [#]')
    plt.ylabel('abs. (real - pred)')
    plt.legend()
    plt.figure()
    plt.scatter(range(len(correct_zeros)), 100*np.array(correct_zeros))
    plt.xlabel('run number [#]')
    plt.ylabel('correct_zeros')
    plt.figure()
    plt.scatter(range(len(correct_non_zeros)), 100 * np.array(correct_non_zeros))
    plt.xlabel('run number [#]')
    plt.ylabel('correct non zeros')

    print("average RMSE for "+str(numRuns)+" runs is:"+str(np.mean(np.array(rmse))))
    print("average accuracy for "+str(numRuns)+" runs is:"+str(100*np.mean(np.array(accuracy))))
    print("average corrected zeros "+str(numRuns)+" runs is:" + str(100*np.mean(np.array(correct_zeros))))
    print("average corrected non zeros for "+str(numRuns)+" runs is:" + str(100*np.mean(np.array(correct_non_zeros))))

    plt.show()
    return




if __name__ == '__main__':
    main()
    print('Done.')