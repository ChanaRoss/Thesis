# mathematical imports
import numpy as np
from sklearn import metrics
import pandas as pd
# graphical imports
from matplotlib import pyplot as plt
import seaborn as sns
# system imports
import pickle
import sys
import os
# pytorch imports
import torch
import torch.utils.data as data
# my file imports
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/')
from LSTM_inputFullGrid_multiClassSmooth import Model, moving_average
from dataLoader_uber import DataSet_oneLSTM_allGrid
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif

sns.set()
np.random.seed(1)

def loadFile(fileName):
    with open(fileName+'.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data


def checkResults(data_net, data_labels, data_real, timeIndexs, xIndexs, yIndexs, filePath, fileName, dataType):
    clr = np.random.rand(3, 3)
    t_size = data_net.shape[0]
    x_size = data_net.shape[1]
    y_size = data_net.shape[2]
    if dataType == 'Real':
        fileName = 'RealInput_'+fileName
    elif dataType == 'Smooth':
        fileName = 'SmoothInput_'+fileName
    for i, x in enumerate(xIndexs):
        for j, y in enumerate(yIndexs):
            fig, ax = plt.subplots(1, 1)
            # ax.plot(timeIndex, data_labels[:, i, j], color='g', marker='.', linestyle='-.', label=dataType + ' label output')
            ax.plot(timeIndexs, data_real[:, i, j],   color='m', marker='o', linestyle='-', label=dataType + ' output')
            ax.plot(timeIndexs, data_net[:, i, j],    color='k', marker='*', linestyle='--',  label='net output')
            ax.set_xlabel('Time')
            ax.set_ylabel('Num Events')
            ax.set_title(fileName + ', (x,y):(' + str(x) + ',' + str(y) + ')')
            plt.legend()
            plt.show()
            # plt.savefig(filePath + 'timeResults_' + fileName + str(x) + 'x_' + str(y) + 'y_' + '.png')
            # plt.close()
    return


def plotSpesificTime(data_net, data_labels, data_real, t, pathName, fileName, dataType, maxTick):
    dataReal = data_real.reshape(data_real.shape[1], data_real.shape[2])
    dataPred = data_net.reshape(data_net.shape[1], data_net.shape[2])

    day              = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week             = np.floor_divide(t, 2 * 24 * 7) + 14  # first week is week 14 of the year
    temp_hour, temp  = np.divmod(t, 2)
    temp_hour        += 8  # data starts from 98 but was normalized to 0
    _, hour          = np.divmod(temp_hour, 24)
    minute           = temp * 30

    dataFixed   = np.zeros_like(dataReal)
    dataFixed   = np.swapaxes(dataReal, 1, 0)
    dataFixed   = np.flipud(dataFixed)

    dataFixedPred = np.zeros_like(dataPred)
    dataFixedPred = np.swapaxes(dataPred, 1, 0)
    dataFixedPred = np.flipud(dataFixedPred)

    f, axes = plt.subplots(1, 2)
    ticksDict = list(range(maxTick+1))

    sns.heatmap(dataFixed,     cbar=False, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[0], cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    sns.heatmap(dataFixedPred, cbar=True, center=1, square=True, vmin=0, vmax=np.max(ticksDict),  ax=axes[1], cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    axes[0].set_title('week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute) + ' , ' + dataType + ' data')
    axes[1].set_title('time- {0}:{1}'.format(hour, minute) + ' , net predicted data')
    # plt.title('time is -  {0}:{1}'.format(hour, minute))
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel('Y axis')
    axes[1].set_xlabel('X axis')
    axes[1].set_ylabel('Y axis')
    plt.savefig(pathName + dataType + '_' + fileName + '_' + str(t) +'.png')
    plt.close()
    return


def plotSpesificTime_allData(data_net, data_real, t, pathName, fileName, dataType, maxTick):
    dataReal    = data_real.reshape(data_real.shape[1], data_real.shape[2])
    dataPred    = data_net.reshape(data_net.shape[1], data_net.shape[2])
    day              = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week             = np.floor_divide(t, 2 * 24 * 7) + 14  # first week is week 14 of the year
    temp_hour, temp  = np.divmod(t, 2)
    temp_hour        += 8  # data starts from 98 but was normalized to 0
    _, hour          = np.divmod(temp_hour, 24)
    minute           = temp * 30

    dataFixed   = np.zeros_like(dataReal)
    dataFixed   = np.swapaxes(dataReal, 1, 0)
    dataFixed   = np.flipud(dataFixed)

    dataFixedPred = np.zeros_like(dataPred)
    dataFixedPred = np.swapaxes(dataPred, 1, 0)
    dataFixedPred = np.flipud(dataFixedPred)


    f, axes = plt.subplots(1, 2)
    ticksDict = list(range(maxTick+1))
    cbar_ax = f.add_axes([.91, .3, .03, .4])
    sns.heatmap(dataFixedPred, cbar=False, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[0], cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    sns.heatmap(dataFixed,     cbar=True, cbar_ax=cbar_ax, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[1], cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))
    f.suptitle('week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute), fontsize=16)
    axes[0].set_title('net-real data')
    axes[1].set_title('real data')
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel('Y axis')
    axes[1].set_xlabel('X axis')
    plt.savefig(pathName + dataType + '_' + fileName + '_' + str(t) +'.png')
    plt.close()
    return


def createTimeGif(net_data, labels_data, real_data, timeIndexs, fileName, pathName, dataType):
    lengthX = net_data.shape[1]
    lengthY = net_data.shape[2]
    maxTick = np.max(real_data).astype(int)
    for i, t in enumerate(timeIndexs):
        temp_net = net_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_real = real_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_label = labels_data[i, :, :].reshape([1, lengthX, lengthY])
        plotSpesificTime(temp_net, temp_label, temp_real, t, pathName, fileName, dataType, maxTick)
    listNames = [dataType + '_' + fileName + '_' + str(t) + '.png' for t in timeIndexs]
    create_gif(pathName, listNames, 1, dataType + '_' + fileName)
    for fileName in listNames:
        os.remove(pathName + fileName)
    return


def createTimeGif_allData(net_data, real_data, timeIndexs, fileName, pathName, dataType):
    lengthX = net_data.shape[1]
    lengthY = net_data.shape[2]
    maxTick = np.max(real_data).astype(int)
    for i, t in enumerate(timeIndexs):
        temp_net    = net_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_real   = real_data[i, :, :].reshape([1, lengthX, lengthY])
        plotSpesificTime_allData(temp_net, temp_real, t, pathName, fileName, dataType, maxTick)
    listNames = [dataType + '_' + fileName + '_' + str(t) + '.png' for t in timeIndexs]
    create_gif(pathName, listNames, 1, dataType + '_' + fileName)
    for fileName in listNames:
        os.remove(pathName + fileName)
    return


def create_bm_seq_input(data, t_index, seq_len, createLabel = True):
    # in this case we assume that the batches are the different grid points
    t_size = data.shape[0]
    x_size = data.shape[1]
    y_size = data.shape[2]
    temp = np.zeros(shape=(seq_len, x_size, y_size))
    if (t_index - seq_len > 0):
        temp = data[t_index - seq_len:t_index, :, :]
    else:
        temp[seq_len - t_index:, :, :] = data[0:t_index, :, :]
    xArr = np.zeros(shape=(x_size * y_size, seq_len))
    if createLabel:
        tempOut = np.zeros(shape=(seq_len, x_size, y_size))
        try:

            if (t_index + 1 <= t_size) and (t_index + 1 - seq_len > 0):
                tempOut = data[t_index + 1 - seq_len: t_index + 1, :, :].reshape(seq_len, x_size, y_size)
            elif (t_index + 1 <= t_size) and (t_index + 1 - seq_len <= 0):
                tempOut[seq_len - t_index - 1:, :, :] = data[0:t_index + 1, :, :]
            elif (t_index + 1 > t_size) and (t_index + 1 - seq_len > 0):
                tempOut[0:seq_len - 1, :, :] = data[t_index + 1 - seq_len: t_index, :, :]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')

        try:
            yArrTemp = tempOut[-1, :, :]
        except:
            print("couldnt take last value of time sequence for output!!!")

        yArr = np.zeros(shape=(x_size*y_size))
    k = 0
    for i in range(x_size):
        for j in range(y_size):
            xArr[k, :] = temp[:, i, j]
            if createLabel:
                yArr[k]    = yArrTemp[i, j]
            k += 1
    # xArr is of shape: [grid id, seq]
    # yArr is of shape: [grid id]
    if not createLabel:
        yArr = []
    return xArr, yArr


def calc_bm_seq_Output_fromData_probability(data_in, class_size, timeIndexs, sequence_size):
    time_size       = timeIndexs.size
    x_size          = data_in.shape[1]
    y_size          = data_in.shape[2]
    grid_size       = x_size*y_size

    net_accuracy        = []
    net_rmse            = []
    correct_non_zeros   = []
    correct_zeros       = []
    numEventsCreated    = []
    numEventsPredicted  = []

    bm_seq_output      = np.zeros((time_size, x_size, y_size))
    label_output    = np.zeros((time_size, x_size, y_size))
    # testOutTemp = []
    # netlabelTemp = []
    # inputTemp    = []
    cdf             = np.zeros(class_size)
    events_mat  = np.zeros(shape=[x_size, y_size, class_size])
    for i, t in enumerate(timeIndexs):
        # print("t:" + str(i) + " out of:" + str(timeIndexs.size))
        for x in range(x_size):
            for y in range(y_size):
                # print("t:" + str(i) + ",x:"+str(x)+", y:"+str(y))
                input, label = create_bm_seq_input(data_in[:, x, y].reshape([-1, 1, 1]), t, sequence_size)
                # input size is: [grid_id, seq_len]
                # label size is: [grid_id]
                for t_seq in range(sequence_size):
                    num_events = int(input[0, t_seq])
                    events_mat[x, y, num_events] += 1
                events_mat[x, y, :] = events_mat[x, y, :]/np.sum(events_mat[x, y, :])
                probOut             = events_mat[x, y, :]
                for j in range(probOut.size):
                    cdf[j] = np.sum(probOut[0:j + 1])
                randNum = np.random.uniform(0, 1)
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdf, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                bm_labelsNp = np.array([numEvents]).reshape([1, 1])
                # testOutTemp.append(testOut.detach().numpy())
                # netlabelTemp.append(net_labelsNp)
                # inputTemp.append(input)
                labelsNp = label
                sizeMat = bm_labelsNp.size
                net_rmse.append(np.sqrt(metrics.mean_squared_error(labelsNp.reshape(-1), bm_labelsNp.reshape(-1))))
                net_accuracy.append(np.sum(labelsNp == bm_labelsNp) / sizeMat)
                sizeMat_zeros = bm_labelsNp[labelsNp == 0].size
                sizeMat_non_zeros = bm_labelsNp[labelsNp != 0].size
                if (sizeMat_non_zeros > 0):
                    correct_non_zeros.append(np.sum(bm_labelsNp[labelsNp != 0] == labelsNp[labelsNp != 0]) / sizeMat_non_zeros)
                if sizeMat_zeros > 0:
                    correct_zeros.append(np.sum(bm_labelsNp[labelsNp == 0] == labelsNp[labelsNp == 0]) / sizeMat_zeros)
                numEventsCreated.append(np.sum(labelsNp))
                numEventsPredicted.append(np.sum(bm_labelsNp))
                bm_seq_output[i, x, y] = bm_labelsNp
                label_output[i, x, y]  = labelsNp

    results = {'accuracy'           : net_accuracy,
               'rmse'               : net_rmse,
               'numCreated'         : numEventsCreated,
               'numPredicted'       : numEventsPredicted,
               'correctedZeros'     : correct_zeros,
               'correctedNonZeros'  : correct_non_zeros}
    return bm_seq_output, label_output, results


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


def main():
    data_path = '/Users/chanaross/dev/Thesis/UberData/'
    data_name = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'

    # network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/uberSingleGridTrain_longSeq/'
    network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/singleGridId_multiClassSmooth/'
    # GPU_results/singleGridId_multiClassSmooth/'
    # network_name = 'smooth_30_seq_50_bs_40_hs_128_lr_0.9_ot_1_wd_0.002_torch.pkl'
    network_name = 'smooth_10_seq_50_bs_40_hs_128_lr_0.9_ot_1_wd_0.002_torch.pkl'
    my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
    sequence_size = my_net.sequence_size

    plot_graph_vs_time = True
    plot_time_gif      = False

    # create dictionary for storing result for each network tested
    results = {'mean RMSE'                      : [],
               'mean accuracy'                  : [],
               'mean zeros accuracy'            : [],
               'mean nonZeros accuracy'         : [],
               'mean smooth RMSE'               : [],
               'mean smooth accuracy'           : [],
               'mean smooth zeros accuracy'     : [],
               'mean smooth nonZeros accuracy'  : []}

    dataInputReal = np.load(data_path + data_name)
    lengthT = dataInputReal.shape[2]
    lengthX = dataInputReal.shape[0]
    lengthY = dataInputReal.shape[1]

    xmin = 0
    xmax = dataInputReal.shape[0]  # 6
    ymin = 0
    ymax = dataInputReal.shape[1]  # 11
    zmin = 0  # np.floor(dataInputReal.shape[2]*0.7).astype(int)
    zmax = dataInputReal.shape[2]
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:zmax]  # shrink matrix size for fast training in order to test model

    class_size = int(np.max(dataInputReal) + 1)
    # dataInputReal = dataInputReal[5:6, 10:11, :]
    # reshape input data for network format -

    dataInputReal = np.swapaxes(dataInputReal, 0, 1)
    dataInputReal = np.swapaxes(dataInputReal, 0, 2)
    # create results index's -
    tmin = 2000
    tmax = 2500
    timeIndexs = np.arange(tmin, tmax, 1).astype(int)
    xIndexs    = np.arange(xmin, xmax, 1).astype(int)
    yIndexs    = np.arange(ymin, ymax, 1).astype(int)
    numRuns = timeIndexs.size
    figPath     = data_path + 'figures/'
    fileName    = str(numRuns) + '_bm_easy_seqBased'

    try:
        smoothParam = my_net.smoothingParam
    except:
        print("no smoothing param, using default 15")
        smoothParam = 15
    # create smooth data -
    dataInputSmooth = moving_average(dataInputReal, smoothParam, axis=0)  # smoothing data so that results are more clear to network

    # calc network output
    # real data -
    bm_output_fromData, label_output_fromData, net_results = calc_bm_seq_Output_fromData_probability(dataInputReal,
                                                                                                      class_size,
                                                                                                      timeIndexs,
                                                                                                      sequence_size)

    bm_output_fromData_smooth, label_output_fromData_smooth, net_results_smooth = calc_bm_seq_Output_fromData_probability(dataInputSmooth,
                                                                                                     class_size,
                                                                                                     timeIndexs,
                                                                                                     sequence_size)

    # plot each grid point vs. time
    if plot_graph_vs_time:
        # real data-
        checkResults(bm_output_fromData, label_output_fromData, dataInputReal[timeIndexs, :, :],
                     timeIndexs, xIndexs, yIndexs, figPath, fileName, 'Real')

        # real data-
        checkResults(bm_output_fromData_smooth, label_output_fromData_smooth, dataInputSmooth[timeIndexs, :, :],
                     timeIndexs, xIndexs, yIndexs, figPath, fileName, 'Smooth')

    # save total results to dictionary for each network tested -
    # real data -
    results['mean RMSE'].append(np.mean(net_results['rmse']))
    results['mean accuracy'].append(100 * np.mean(net_results['accuracy']))
    results['mean nonZeros accuracy'].append(100 * np.mean(net_results['correctedNonZeros']))
    results['mean zeros accuracy'].append(100 * np.mean(net_results['correctedZeros']))
    # smooth data -
    results['mean smooth RMSE'].append(np.mean(net_results_smooth['rmse']))
    results['mean smooth accuracy'].append(100 * np.mean(net_results_smooth['accuracy']))
    results['mean smooth nonZeros accuracy'].append(100 * np.mean(net_results_smooth['correctedNonZeros']))
    results['mean smooth zeros accuracy'].append(100 * np.mean(net_results_smooth['correctedZeros']))

    if plot_time_gif:
        # create gif for results -
        # all data -
        createTimeGif_allData(bm_output_fromData, dataInputReal[timeIndexs, :, :], timeIndexs, fileName, figPath, 'Real')

        # # real data
        # createTimeGif(net_output_fromData, label_output_fromData, dataInputReal[timeIndexs, :, :], timeIndexs, fileName, figPath, 'Real')

    # print all results -
    # real data -
    print("average RMSE for " + str(numRuns) + " runs is:" + str(np.mean(net_results['rmse'])))
    print("average accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['accuracy'])))
    print("average corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['correctedZeros'])))
    print("average corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['correctedNonZeros'])))
    # smooth data -
    print("average smooth RMSE for " + str(numRuns) + " runs is:" + str(np.mean(net_results_smooth['rmse'])))
    print("average smooth accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results_smooth['accuracy'])))
    print("average smooth corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results_smooth['correctedZeros'])))
    print("average smooth corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results_smooth['correctedNonZeros'])))

    return




if __name__ == '__main__':
    main()
    print('Done.')

