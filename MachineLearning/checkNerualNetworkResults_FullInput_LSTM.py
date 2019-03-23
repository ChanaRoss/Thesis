# mathematical imports -
import numpy as np
from matplotlib import  pyplot as plt
from sklearn import metrics
from math import sqrt
import seaborn as sns
sns.set()
import torch

# load network imports -
import os
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/')
from LSTM_inputFullGrid import Model
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif


def createRealEventsUberML_network(eventMatrix, startTime, endTime):
    firstTime = startTime
    if (endTime - startTime==0):
        numTimeSteps = 1
    else:
        numTimeSteps = endTime - startTime

    realMatOut = eventMatrix[firstTime: firstTime + numTimeSteps, :, :]
    return realMatOut


def getInputMatrixToNetwork(previousMat):
    # previousMat is of shape : [1, seq_len , size_x, size_y]
    lenSeqIn = previousMat.shape[0]
    lengthX = previousMat.shape[1]
    lengthY = previousMat.shape[2]
    xArr = np.zeros(shape=(1, lenSeqIn, lengthX * lengthY))
    k = 0
    for i in range(lengthX):
        for j in range(lengthY):
            xArr[0, :, k] = previousMat[:, i, j]
            k += 1

    if torch.cuda.is_available():
        xTensor = torch.Tensor(xArr).cuda()
    else:
        xTensor = torch.Tensor(xArr)
    # xTensor is of shape: [seq, grid_size]
    return xTensor


def createEventDistributionUber(previousEventMatrix, my_net, eventTimeWindow, startTime, endTime):
    """
    this function calculates future events based on cnn lstm network
    :param previousEventMatrix: event matrix of previous events
    :param my_net: learned network
    :param eventTimeWindow: time each event is opened (for output)
    :param startTime: start time from which events are created
    :param endTime: end time to create events (start and end time define the output sequence length)
    :return: eventPos, eventTimeWindow, outputEventMat
    """
    # previousEventMatrix is of size: [seq_len, x_size, y_size]

    if endTime - startTime == 0:  # should output one prediction
        out_seq = 1
    else:
        out_seq = endTime - startTime
    x_size = previousEventMatrix.shape[1]
    y_size = previousEventMatrix.shape[2]
    netEventOut = torch.zeros([out_seq, x_size, y_size])
    for seq in range(out_seq):
        tempEventMat = previousEventMatrix
        input = getInputMatrixToNetwork(previousEventMatrix)
        testOut = my_net.forward(input)
        t = torch.autograd.Variable(torch.Tensor([0.5]))  # threshold
        netEventOut = (testOut > t).float() * 1
        netEventOut = netEventOut.reshape(shape=(1, x_size, y_size))
        previousEventMatrix[0:-1, :, :]     = tempEventMat[1:, :, :]
        previousEventMatrix[-1, :, :]       = netEventOut[seq, :, :]
    # in the end netEventOut is a matrix of size [out_seq_len, size_x, size_y]
    eventPos    = []
    eventTimes  = []
    for t in range(out_seq):
        for x in range(x_size):
            for y in range(y_size):
                numEvents = netEventOut[t, x, y]
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                #for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t+startTime)
    eventsPos  = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsPos, eventsTimeWindow, netEventOut.detach().numpy()

def getPreviousEventMat(dataInputReal, start_time, in_seq_len = 5):
    lastPreviousTime = start_time
    previousEventMatrix = np.zeros(shape=(in_seq_len, dataInputReal.shape[1], dataInputReal.shape[2]))
    if lastPreviousTime - in_seq_len >= 0:  # there are enough previous events known to system
        previousEventMatrix = dataInputReal[lastPreviousTime-in_seq_len:lastPreviousTime, :, :]
    else: # need to pad results
        previousEventMatrix[in_seq_len-lastPreviousTime:, :, :] = dataInputReal
    return previousEventMatrix


def plotSpesificTime(dataReal, dataPred, t, fileName):
    dataReal = dataReal.reshape(dataReal.shape[1], dataReal.shape[2])
    dataPred = dataPred.reshape(dataPred.shape[1], dataPred.shape[2])

    day         = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week        = np.floor_divide(t, 2 * 24 * 7) + 14  # first week is week 14 of the year
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
    ticksDict = list(range(2))

    sns.heatmap(dataFixed, cbar = False, center = 1, square=True, vmin = 0, vmax = 1, ax=axes[0], cmap = 'CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    sns.heatmap(dataFixedPred, cbar=True, center=1, square=True, vmin=0, vmax=1, ax=axes[1], cmap='CMRmap_r',cbar_kws=dict(ticks=ticksDict))
    axes[0].set_title('week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute) + ' , Real data')
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
    # network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/limitedZero_500grid/'
    # network_name = 'gridSize11_epoch86_batch35_torch.pkl'

    network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/gridInput_LSTM/'
    network_name = 'gridSize11_epoch4_batch140_torch.pkl'

    data_path    = '/Users/chanaross/dev/Thesis/UberData/'
    data_name    = '3D_allDataLatLonCorrected_binaryClass_500gridpickle_30min.p'


    # network_name = 'gridSize20_epoch608_batch9_torch.pkl'
    # data_path = '/Users/chanaross/dev/Thesis/UberData/'
    # data_name = '3D_allDataLatLonCorrected_500gridpickle_30min.p'


    dataInputReal     = np.load(data_path + data_name)

    my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
    my_net.eval()

    xmin = 0
    xmax = dataInputReal.shape[0]
    ymin = 0
    ymax = dataInputReal.shape[1]
    zmin = 48
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:]   #shrink matrix size for fast training in order to test model
    # dataInputReal[dataInputReal > 1] = 1

    # reshape input data for network format -
    lengthT = dataInputReal.shape[2]
    lengthX = dataInputReal.shape[0]
    lengthY = dataInputReal.shape[1]
    dataInputReal = np.swapaxes(dataInputReal, 0, 1)
    dataInputReal = np.swapaxes(dataInputReal, 0, 2)
    # dataInputReal       = dataInputReal.reshape(lengthT, lengthX, lengthY)
    accuracy            = []
    rmse                = []
    numEventsCreated    = []
    numEventsPredicted  = []
    correct_non_zeros   = []
    correct_zeros       = []
    timeOut             = []
    figPath = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/gridInput_LSTM/figures/'
    numRuns = 10
    fileName = '500grid_30min_binary_network_results_'+str(numRuns)
    for i in range(numRuns):
        print("run num:"+str(i))
        # start_time = i+200
        start_time = np.random.randint(10, dataInputReal.shape[0] - 10)
        timeOut.append(start_time)
        end_time   = start_time + 0
        realMatOut = createRealEventsUberML_network(dataInputReal, start_time, end_time)
        previousEventMatrix = getPreviousEventMat(dataInputReal, start_time, my_net.sequence_size)
        eventsPos, eventsTimeWindow, netEventOut = createEventDistributionUber(previousEventMatrix, my_net, 3, start_time, end_time)

        sizeMat = netEventOut.size
        rmse.append(sqrt(metrics.mean_squared_error(realMatOut.reshape(-1), netEventOut.reshape(-1))))
        accuracy.append(np.sum(realMatOut == netEventOut) / (sizeMat))
        sizeMat_zeros = netEventOut[realMatOut == 0].size
        sizeMat_non_zeros = netEventOut[realMatOut != 0].size
        if (sizeMat_non_zeros>0):
            correct_non_zeros.append(np.sum(netEventOut[realMatOut != 0] == realMatOut[realMatOut != 0]) / sizeMat_non_zeros)
        if sizeMat_zeros>0:
            correct_zeros.append(np.sum(netEventOut[realMatOut == 0] == realMatOut[realMatOut == 0]) / sizeMat_zeros)
        plotSpesificTime(realMatOut, netEventOut, start_time, figPath + fileName)
        numEventsCreated.append(np.sum(realMatOut))
        numEventsPredicted.append(np.sum(netEventOut))

        # realMatOut[realMatOut > 1] = 1
        # distMatOut[distMatOut > 1] = 1
        # accuracy1.append(np.sum(np.sum(realMatOut == distMatOut)/(realMatOut.shape[0]*realMatOut.shape[1])))
        # if (realMatOut[realMatOut!=0].size >0):
        #     non_zero_accuracy1.append(np.sum(np.sum(realMatOut[realMatOut != 0] == distMatOut[realMatOut != 0]))/(realMatOut[realMatOut != 0].size))
        #
        # if (distMatOut[distMatOut!=0].size >0):
        #     non_zero_accuracy1_dist.append(np.sum(np.sum(realMatOut[distMatOut != 0] == distMatOut[distMatOut != 0]))/(realMatOut[distMatOut != 0].size))

    listNames = [fileName + '_' + str(t) + '.png' for t in timeOut]
    create_gif(figPath, listNames, 1, fileName)

    plt.scatter(range(len(accuracy)), 100 * np.array(accuracy))
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
    plt.scatter(range(len(numEventsCreated)), np.abs(np.array(numEventsCreated) - np.array(numEventsPredicted)),label="difference between prediction and real")
    plt.xlabel('run number [#]')
    plt.ylabel('abs. (real - pred)')
    plt.legend()
    plt.figure()
    plt.scatter(range(len(correct_zeros)), 100 * np.array(correct_zeros))
    plt.xlabel('run number [#]')
    plt.ylabel('correct_zeros')
    plt.figure()
    plt.scatter(range(len(correct_non_zeros)), 100 * np.array(correct_non_zeros))
    plt.xlabel('run number [#]')
    plt.ylabel('correct non zeros')

    print("average RMSE for " + str(numRuns) + " runs is:" + str(np.mean(np.array(rmse))))
    print("average accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(accuracy))))
    print("average corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_zeros))))
    print("average corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_non_zeros))))

    plt.show()
    return





if __name__ == '__main__':
    main()
    print('Done.')