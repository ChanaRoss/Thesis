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
from CNN_LSTM_NeuralNetOrigV2 import Model


def createRealEventsUberML_network(eventMatrix, startTime, endTime):
    firstTime = startTime
    if (endTime - startTime==0):
        numTimeSteps = 1
    else:
        numTimeSteps = endTime - startTime

    realMatOut = eventMatrix[ firstTime: firstTime + numTimeSteps, :, :]
    return realMatOut


def getInputMatrixToNetwork(previousMat, sizeCnn):
    # previousMat is of shape : [seq_len , size_x, size_y]
    lenSeqIn = previousMat.shape[0]
    lengthX = previousMat.shape[1]
    lengthY = previousMat.shape[2]
    temp2 = np.zeros(shape=(1, lenSeqIn, sizeCnn, sizeCnn, lengthX * lengthY))
    tempPadded = np.zeros(shape=(lenSeqIn, lengthX + sizeCnn, lengthY + sizeCnn))
    tempPadded[:, sizeCnn: sizeCnn + lengthX, sizeCnn: sizeCnn + lengthY] = previousMat
    k = 0
    for i in range(lengthX):
        for j in range(lengthY):
            try:
                temp2[0, :, :, :, k] = tempPadded[:, i:i + sizeCnn, j: j + sizeCnn]
            except:
                print("couldnt create input for cnn ")
            k += 1
    xArr = temp2

    if torch.cuda.is_available():
        xTensor = torch.Tensor(xArr).cuda()
    else:
        xTensor = torch.Tensor(xArr)
    # xTensor is of shape: [grid id, seq, x_cnn, y_cnn]
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
        input = getInputMatrixToNetwork(previousEventMatrix, 9)
        k = 0
        for x in range(x_size):
            for y in range(y_size):  # calculate output for each grid_id
                testOut = my_net.forward(input[:, :, :, :, k])
                _, netEventOut[seq, x, y]   = torch.max(torch.exp(testOut.data), 1)
                k += 1
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
    lastPreviousTime = start_time - 1
    previousEventMatrix = np.zeros(shape=(5, dataInputReal.shape[1], dataInputReal.shape[2]))
    if lastPreviousTime - in_seq_len >= 0:  # there are enough previous events known to system
        previousEventMatrix = dataInputReal[lastPreviousTime-in_seq_len:lastPreviousTime, :, :]
    else: # need to pad results
        previousEventMatrix[in_seq_len-lastPreviousTime:, :, :] = dataInputReal
    return previousEventMatrix


def main():
    network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/limitedZero_w0p05_v2/'
    network_name = 'gridSize20_epoch60_batch24_torch.pkl'
    data_path    = '/Users/chanaross/dev/Thesis/UberData/'
    data_name    = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_allData.p'
    dataInputReal     = np.load(data_path + data_name)

    my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
    my_net.eval()

    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 40
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, 24000:32000]  #shrink matrix size for fast training in order to test model
    # reshape input data for network format -
    lengthT = dataInputReal.shape[2]
    lengthX = dataInputReal.shape[0]
    lengthY = dataInputReal.shape[1]
    dataInputReal = dataInputReal.reshape(lengthT, lengthX, lengthY)

    accuracy  = []
    accuracy1 = []
    rmse      = []
    numEventsCreated   = []
    numEventsPredicted = []
    for i in range(5):
        print("run num:"+str(i))
        start_time = np.random.randint(10, dataInputReal.shape[0] - 10)
        end_time   = start_time + 0
        realMatOut = createRealEventsUberML_network(dataInputReal, start_time, end_time)
        previousEventMatrix = getPreviousEventMat(dataInputReal, start_time, 5)
        eventsPos, eventsTimeWindow, netEventOut = createEventDistributionUber(previousEventMatrix, my_net, 3, start_time, end_time)
        rmse.append(sqrt(metrics.mean_squared_error(realMatOut.reshape(-1), netEventOut.reshape(-1))))
        accuracy.append(np.sum(realMatOut == netEventOut) / (realMatOut.shape[0] * realMatOut.shape[1] * realMatOut.shape[2]))
        realMatOut[realMatOut > 1] = 1
        netEventOut[netEventOut > 1] = 1
        accuracy1.append(np.sum(np.sum(realMatOut == netEventOut) / (realMatOut.shape[0] * realMatOut.shape[1] * realMatOut.shape[2])))
        numEventsCreated.append(np.sum(realMatOut))
        numEventsPredicted.append(np.sum(netEventOut))
        print("run num:" + str(i) +", finished")
    plt.scatter(range(len(accuracy)), 100 * np.array(accuracy))
    plt.xlabel('run number [#]')
    plt.ylabel('accuracy [%]')
    plt.figure()
    plt.scatter(range(len(rmse)), np.array(rmse))
    plt.xlabel('run number [#]')
    plt.ylabel('RMSE')
    plt.figure()
    plt.scatter(range(len(numEventsCreated)), np.array(numEventsCreated), label = "num real events")
    plt.scatter(range(len(numEventsPredicted)), np.array(numEventsPredicted), label = "num predicted")
    plt.xlabel('run number [#]')
    plt.ylabel('num events created')
    plt.legend()
    plt.figure()
    plt.scatter(range(len(accuracy1)), 100 * np.array(accuracy1))
    print("average RMSE for 300 runs is:" + str(np.mean(np.array(rmse))))
    print("average accuracy for 300 runs is:" + str(np.mean(np.array(accuracy))))
    print("average corrected accuracy for 300 runs is:" + str(np.mean(np.array(accuracy1))))

    plt.show()
    return





if __name__ == '__main__':
    main()
    print('Done.')