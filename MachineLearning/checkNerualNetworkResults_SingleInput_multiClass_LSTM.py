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
from LSTM_inputFullGrid_multiClassSmooth import Model
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from dataLoader_uber import createDescentLabels
from createGif import create_gif
import pandas as pd
from sklearn.metrics import confusion_matrix

np.random.seed(1)
sns.set(rc={'figure.figsize':(14, 9)})


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
        testOut = testOut.view(1, my_net.class_size, x_size*y_size)
        _, netEventOut = torch.max(torch.exp(testOut.data), 1)
        netEventOut = netEventOut.view(1, x_size, y_size)
        previousEventMatrix[0:-1, :, :]     = tempEventMat[1:, :, :]
        previousEventMatrix[-1, :, :]       = netEventOut[seq, :, :].detach().numpy()
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
        previousEventMatrix[in_seq_len-lastPreviousTime:, :, :] = dataInputReal[:lastPreviousTime, :, :]
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
    ticksDict = list(range(21))

    sns.heatmap(dataFixed, cbar = False, center = 1, square=True, vmin = 0, vmax = 20, ax=axes[0], cmap = 'CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    sns.heatmap(dataFixedPred, cbar=True, center=1, square=True, vmin=0, vmax=20, ax=axes[1], cmap='CMRmap_r',cbar_kws=dict(ticks=ticksDict))
    axes[0].set_title('week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute) + ' , Real data')
    axes[1].set_title('time- {0}:{1}'.format(hour, minute) + ' , Predicted data')
    # plt.title('time is -  {0}:{1}'.format(hour, minute))
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel('Y axis')
    axes[1].set_xlabel('X axis')
    axes[1].set_ylabel('Y axis')
    plt.savefig(fileName + '_' + str(t) +'.png')
    plt.close()
    return


def plotEpochGraphs(my_net, filePath, fileName):
    f, ax = plt.subplots(2,1)
    ax[0].plot(range(len(my_net.accVecTrain)), np.array(my_net.accVecTrain), label = "Train accuracy")
    ax[0].plot(range(len(my_net.accVecTest)), np.array(my_net.accVecTest), label="Test accuracy")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Accuracy [%]')

    ax[1].plot(range(len(my_net.lossVecTrain)), np.array(my_net.lossVecTrain), label="Train Loss")
    ax[1].plot(range(len(my_net.lossVecTest)), np.array(my_net.lossVecTest), label="Test Loss")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.legend()
    plt.savefig(filePath + 'lossResults_' + fileName + '.png')
    plt.close()
    # plt.show()
    return

def plot_confusion_matrix(y_true, y_pred, classes,
                          t, fileLoc, fileName,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    day = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week = np.floor_divide(t, 2 * 24 * 7) + 14  # first week is week 14 of the year
    hour, temp = np.divmod(t, 2)
    hour += 8  # data starts from 98 but was normalized to 0
    _, hour = np.divmod(hour, 24)
    minute = temp * 30
    title = 'week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute) + ' , Confusion matrix'
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)
        cm = np.nan_to_num(cm)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    num_ticks = len(classes)
    # the index of the position of yticks
    yticks = np.linspace(0, len(classes) - 1, num_ticks, dtype=np.int)
    # the content of labels of these yticks
    yticklabels = [classes[idx].astype(int) for idx in yticks]
    ax = sns.heatmap(cm, cmap=cmap, annot=True, yticklabels=yticklabels, xticklabels= yticklabels)
    ax.set_yticks(yticks+0.5)
    ax.set_xticks(yticks+0.5)
    ax.set_title(title)
    ax.set_xlabel('Real Classification')
    ax.set_ylabel('Predicted Classification')
    plt.savefig(fileLoc + 'ConfMat_'+fileName + '_' + str(t) +'.png')
    plt.close()
    return

def moving_average(data_set, periods=3, axis = 2):
    cumsum = np.cumsum(data_set, axis = axis)
    averageRes =  (cumsum[:, :, periods:] - cumsum[:, :, :-periods]) / float(periods)
    return np.floor(averageRes)

    # weights = np.ones(periods) / periods
    # convRes = np.convolve(data_set, weights, mode='valid')
    # return np.floor(convRes)


def main():
    network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/singleGridId_multiClassSmooth/'
    # network_names = [f for f in os.listdir(network_path) if (f.endswith('.pkl'))]
    network_names = ['smooth_20_seq_30_bs_40_hs_128_lr_0.5_ot_1_wd_0.002_torch.pkl']
    data_path    = '/Users/chanaross/dev/Thesis/UberData/'
    data_name    = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'


    # dataInputReal       = dataInputReal.reshape(lengthT, lengthX, lengthY)
    results = {'networkName'            : [],
               'mean RMSE'              : [],
               'mean accuracy'          : [],
               'mean zeros accuracy'    : [],
               'mean nonZeros accuracy' : []}
    dataInputReal = np.load(data_path + data_name)
    xmin = 0
    xmax = dataInputReal.shape[0]
    ymin = 0
    ymax = dataInputReal.shape[1]
    zmin = 48
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:]  # shrink matrix size for fast training in order to test model
    # numRuns = 20
    # timeIndexs = [np.random.randint(50, dataInputReal.shape[2] - 50) for i in range(numRuns)]
    # print(timeIndexs)
    timeIndexs = np.arange(200, 500, 1).astype(int)

    numRuns = timeIndexs.size
    for network_name in network_names:
        print("network:" + network_name.replace('.pkl', ''))
        my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
        my_net.eval()
        dataInputReal = np.load(data_path + data_name)
        xmin = 0
        xmax = dataInputReal.shape[0]
        ymin = 0
        ymax = dataInputReal.shape[1]
        zmin = 48
        dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:]  # shrink matrix size for fast training in order to test model
        dataInputReal = dataInputReal[5:6, 10:11, :]
        try:
            smoothParam = my_net.smoothingParam
        except:
            print("no smoothing param, using default 15")
            smoothParam = 15
        dataInputSmooth = moving_average(dataInputReal, smoothParam)  # smoothing data so that results are more clear to network

        # dataInputReal[dataInputReal > 1] = 1
        # reshape input data for network format -
        lengthT = dataInputReal.shape[2]
        lengthX = dataInputReal.shape[0]
        lengthY = dataInputReal.shape[1]
        dataInputReal = np.swapaxes(dataInputReal, 0, 1)
        dataInputReal = np.swapaxes(dataInputReal, 0, 2)
        dataInputSmooth = np.swapaxes(dataInputSmooth, 0, 1)
        dataInputSmooth = np.swapaxes(dataInputSmooth, 0, 2)

        accuracy            = []
        rmse                = []
        numEventsCreated    = []
        numEventsPredicted  = []
        correct_non_zeros   = []
        correct_zeros       = []
        timeOut             = []
        accuracySm          = []
        correct_non_zerosSm = []
        correct_zerosSm     = []
        numEventsCreatedSm  = []
        numEventsPredictedSm= []
        rmseSm              = []
        figPath = network_path + 'figures/'
        fileName = str(numRuns) + network_name.replace('.pkl', '')
        plotEpochGraphs(my_net, figPath, fileName)
        y_pred = []
        y_true = []
        y_trueSm = []
        y_predSm = []
        for i in timeIndexs:
            # print("run num:"+str(i))
            # start_time = i+2000
            start_time = i
            start_timeSm = i+smoothParam
            timeOut.append(start_time)
            end_time   = start_time + 0
            end_timeSm = start_timeSm + 0
            realMatOut   = createRealEventsUberML_network(dataInputReal, start_time, end_time)
            realMatOutSm = createRealEventsUberML_network(dataInputSmooth, start_timeSm, end_timeSm)
            previousEventMatrix = getPreviousEventMat(dataInputReal, start_time, my_net.sequence_size)
            previousEventMatrixSm = getPreviousEventMat(dataInputSmooth, start_timeSm, my_net.sequence_size)
            eventsPos, eventsTimeWindow, netEventOut = createEventDistributionUber(previousEventMatrix, my_net, 3, start_time, end_time)
            _, _, netEventOutSm = createEventDistributionUber(previousEventMatrixSm, my_net, 3, start_time, end_time)

            sizeMat = netEventOut.size
            rmse.append(sqrt(metrics.mean_squared_error(realMatOut.reshape(-1), netEventOut.reshape(-1))))
            rmseSm.append(sqrt(metrics.mean_squared_error(realMatOutSm.reshape(-1), netEventOut.reshape(-1))))
            accuracy.append(np.sum(realMatOut == netEventOut) / (sizeMat))
            accuracySm.append(np.sum(realMatOutSm == netEventOutSm)/ sizeMat)
            sizeMat_zeros = netEventOut[realMatOut == 0].size
            sizeMat_non_zeros = netEventOut[realMatOut != 0].size
            sizeMat_non_zerosSm = netEventOut[realMatOutSm != 0].size
            sizeMat_zerosSm = netEventOut[realMatOutSm == 0].size
            if (sizeMat_non_zeros>0):
                correct_non_zeros.append(np.sum(netEventOut[realMatOut != 0] == realMatOut[realMatOut != 0]) / sizeMat_non_zeros)
            if sizeMat_zeros>0:
                correct_zeros.append(np.sum(netEventOut[realMatOut == 0] == realMatOut[realMatOut == 0]) / sizeMat_zeros)

            if (sizeMat_non_zerosSm>0):
                correct_non_zerosSm.append(np.sum(netEventOutSm[realMatOutSm != 0] == realMatOutSm[realMatOutSm != 0]) / sizeMat_non_zerosSm)
            if sizeMat_zerosSm>0:
                correct_zerosSm.append(np.sum(netEventOutSm[realMatOutSm == 0] == realMatOutSm[realMatOutSm == 0]) / sizeMat_zerosSm)
            numEventsCreated.append(np.sum(realMatOut))
            numEventsPredicted.append(np.sum(netEventOut))
            numEventsPredictedSm.append(np.sum(netEventOutSm))
            numEventsCreatedSm.append(np.sum(realMatOutSm))

            y_true.append(realMatOut.reshape(-1))
            y_pred.append(netEventOut.reshape(-1))
            y_trueSm.append(realMatOutSm.reshape(-1))
            y_predSm.append(netEventOutSm.reshape(-1))

        y_trueAr = np.array(y_true)
        y_predAr = np.array(y_pred)
        y_trueSmAr = np.array(y_trueSm)
        y_predSmAr = np.array(y_predSm)
        plt.plot(timeIndexs, y_trueAr, label='true output', linewidth=2, color = 'b')
        plt.plot(timeIndexs, y_trueSmAr, label='true smooth output', linewidth=2, color ='r')
        plt.plot(timeIndexs, y_predAr, label='predicted output', linewidth=2, color = 'k')
        plt.plot(timeIndexs, y_predSmAr, label='predicted from smooth output', linewidth=2, color ='m')
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('num events')
        plt.show()
        plt.savefig(figPath + 'Results_' + fileName + '.png')
        plt.close()
        # plt.scatter(range(len(accuracy)), 100 * np.array(accuracy))
        # plt.xlabel('run number [#]')
        # plt.ylabel('accuracy [%]')
        # plt.figure()
        # plt.scatter(range(len(rmse)), np.array(rmse))
        # plt.xlabel('run number [#]')
        # plt.ylabel('RMSE')
        # plt.figure()
        # plt.scatter(range(len(numEventsCreated)), np.array(numEventsCreated), label="num real events")
        # plt.scatter(range(len(numEventsPredicted)), np.array(numEventsPredicted), label="num predicted")
        # plt.xlabel('run number [#]')
        # plt.ylabel('num events created')
        # plt.legend()
        # plt.figure()
        # plt.scatter(range(len(numEventsCreated)), np.abs(np.array(numEventsCreated) - np.array(numEventsPredicted)),label="difference between prediction and real")
        # plt.xlabel('run number [#]')
        # plt.ylabel('abs. (real - pred)')
        # plt.legend()
        # plt.figure()
        # plt.scatter(range(len(correct_zeros)), 100 * np.array(correct_zeros))
        # plt.xlabel('run number [#]')
        # plt.ylabel('correct_zeros')
        # plt.figure()
        # plt.scatter(range(len(correct_non_zeros)), 100 * np.array(correct_non_zeros))
        # plt.xlabel('run number [#]')
        # plt.ylabel('correct non zeros')

        print("average RMSE for " + str(numRuns) + " runs is:" + str(np.mean(np.array(rmse))))
        print("average accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(accuracy))))
        print("average corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_zeros))))
        print("average corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_non_zeros))))

        print("smooth average RMSE for " + str(numRuns) + " runs is:" + str(np.mean(np.array(rmseSm))))
        print("smooth average accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(accuracySm))))
        print("smooth average corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_zerosSm))))
        print("smooth average corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(np.array(correct_non_zerosSm))))


        results['networkName'].append(network_name.replace('.pkl', ''))
        results['mean RMSE'].append(np.mean(np.array(rmse)))
        results['mean accuracy'].append(100 * np.mean(np.array(accuracy)))
        results['mean nonZeros accuracy'].append(100 * np.mean(np.array(correct_non_zeros)))
        results['mean zeros accuracy'].append(100 * np.mean(np.array(correct_zeros)))
    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(network_path + 'results.csv')
    plt.show()
    return





if __name__ == '__main__':
    main()
    print('Done.')