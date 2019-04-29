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


def checkNetwork(data_net, data_labels, data_real, timeIndexs, xIndexs, yIndexs, filePath, fileName, dataType):
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


def plotSpesificTime_allData(data_net, data_net_smooth, data_smooth, data_real, t, pathName, fileName, dataType, maxTick):
    dataReal    = data_real.reshape(data_real.shape[1], data_real.shape[2])
    dataPred    = data_net.reshape(data_net.shape[1], data_net.shape[2])
    dataPredSmooth = data_net_smooth.reshape(data_net_smooth.shape[1], data_net_smooth.shape[2])
    dataSmooth  = data_smooth.reshape(data_smooth.shape[1], data_smooth.shape[2])
    day              = np.floor_divide(t, 2 * 24) + 1  # sunday is 1
    week             = np.floor_divide(t, 2 * 24 * 7) + 14  # first week is week 14 of the year
    temp_hour, temp  = np.divmod(t, 2)
    temp_hour        += 8  # data starts from 98 but was normalized to 0
    _, hour          = np.divmod(temp_hour, 24)
    minute           = temp * 30

    dataFixed   = np.zeros_like(dataReal)
    dataFixed   = np.swapaxes(dataReal, 1, 0)
    dataFixed   = np.flipud(dataFixed)

    dataFixedSmooth = np.zeros_like(dataSmooth)
    dataFixedSmooth = np.swapaxes(dataSmooth, 1, 0)
    dataFixedSmooth = np.flipud(dataFixedSmooth)

    dataFixedPred = np.zeros_like(dataPred)
    dataFixedPred = np.swapaxes(dataPred, 1, 0)
    dataFixedPred = np.flipud(dataFixedPred)

    dataFixedPredSmooth = np.zeros_like(dataPredSmooth)
    dataFixedPredSmooth = np.swapaxes(dataPredSmooth, 1, 0)
    dataFixedPredSmooth = np.flipud(dataFixedPredSmooth)

    f, axes = plt.subplots(1, 4)
    ticksDict = list(range(maxTick+1))

    sns.heatmap(dataFixedPred, cbar=False, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[0],cmap='CMRmap_r')
    sns.heatmap(dataFixed,     cbar=False, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[1], cmap='CMRmap_r')
    sns.heatmap(dataFixedSmooth, cbar=False, center=1, square=True, vmin=0, vmax=np.max(ticksDict),  ax=axes[2], cmap='CMRmap_r')
    sns.heatmap(dataFixedPredSmooth, cbar=True, center=1, square=True, vmin=0, vmax=np.max(ticksDict), ax=axes[3],cmap='CMRmap_r', cbar_kws=dict(ticks=ticksDict))

    f.suptitle('week- {0}, day- {1},time- {2}:{3}'.format(week, day, hour, minute), fontsize=16)
    axes[0].set_title('net-real data')
    axes[1].set_title('real data')
    axes[2].set_title('smooth data')
    axes[3].set_title('net-smooth data')
    axes[0].set_xlabel('X axis')
    axes[0].set_ylabel('Y axis')
    axes[1].set_xlabel('X axis')
    axes[2].set_xlabel('X axis')
    axes[3].set_xlabel('X axis')
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


def createTimeGif_allData(net_data, net_data_smooth, smooth_data, real_data, timeIndexs, fileName, pathName, dataType):
    lengthX = net_data.shape[1]
    lengthY = net_data.shape[2]
    maxTick = np.max(real_data).astype(int)
    for i, t in enumerate(timeIndexs):
        temp_net    = net_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_real   = real_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_smooth = smooth_data[i, :, :].reshape([1, lengthX, lengthY])
        temp_net_smooth = net_data_smooth[i, :, :].reshape([1, lengthX, lengthY])
        plotSpesificTime_allData(temp_net, temp_net_smooth, temp_smooth, temp_real, t, pathName, fileName, dataType, maxTick)
    listNames = [dataType + '_' + fileName + '_' + str(t) + '.png' for t in timeIndexs]
    create_gif(pathName, listNames, 1, dataType + '_' + fileName)
    for fileName in listNames:
        os.remove(pathName + fileName)
    return

def create_net_input_allGrid(data, t_index, seq_len, createLabel = True):
    # in this case we assume that the batches are the different grid points
    t_size = data.shape[0]
    x_size = data.shape[1]
    y_size = data.shape[2]
    temp = np.zeros(shape=(seq_len, x_size, y_size))
    if (t_index - seq_len > 0):
        temp = data[t_index - seq_len:t_index, :, :]
    else:
        temp[seq_len - t_index:, :, :] = data[0:t_index, :, :]
    xArr = np.zeros(shape=(x_size * y_size, 1, seq_len))
    if createLabel:
        tempOut = np.zeros(shape=(seq_len, x_size, y_size))
        try:

            if (t_index + 1 <= t_size) and (t_index + 1 - seq_len > 0):
                tempOut = data[t_index + 1 - seq_len: t_index + 1, :, :].reshape(seq_len, x_size, y_size)
            elif (t_index + 1 <= t_size) and (t_index + 1 - seq_len <= 0):
                tempOut[seq_len - t_index - 1:, :, :] = data[0:t_index + 1, :, :]
            elif (t_index + 1 > t_size) and (t_index + 1 - seq_len > 0):
                tempOut[0:seq_len - 1, :, :] = data[t_index + 1 - seq_len: t_index, :,
                                               :]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')

        try:
            yArrTemp = tempOut[-1, :, :]
        except:
            print("couldnt take last value of time sequence for output!!!")

        yArr = np.zeros(shape=(x_size * y_size, 1))
    k = 0
    for i in range(x_size):
        for j in range(y_size):
            xArr[k, 0, :] = temp[:, i, j]
            if createLabel:
                yArr[k, 0] = yArrTemp[i, j]
            k += 1
    if torch.cuda.is_available():
        xTensor = torch.Tensor(xArr).cuda()
        if createLabel:
            yTensor = torch.Tensor(yArr).type(torch.cuda.LongTensor)
    else:
        xTensor = torch.Tensor(xArr)
        if createLabel:
            yTensor = torch.Tensor(yArr).type(torch.long)
    # xTensor is of shape: [grid id, 1, seq]
    # yTensor is of shape: [grid id, 1]
    if not createLabel:
        yTensor = []
    return xTensor, yTensor


def create_net_input(data, t_index, seq_len, createLabel = True):
    # in this case we assume that the batches are the different grid points
    t_size = data.shape[0]
    x_size = data.shape[1]
    y_size = data.shape[2]
    temp = np.zeros(shape=(seq_len, x_size, y_size))
    if (t_index - seq_len > 0):
        temp = data[t_index - seq_len:t_index, :, :]
    else:
        temp[seq_len - t_index:, :, :] = data[0:t_index, :, :]
    xArr = np.zeros(shape=(1, x_size * y_size, seq_len))
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

        yArr = np.zeros(shape=(1, x_size*y_size))
    k = 0
    for i in range(x_size):
        for j in range(y_size):
            xArr[0, k, :] = temp[:, i, j]
            if createLabel:
                yArr[0, k]    = yArrTemp[i, j]
            k += 1
    if torch.cuda.is_available():
        xTensor = torch.Tensor(xArr).cuda()
        if createLabel:
            yTensor = torch.Tensor(yArr).type(torch.cuda.LongTensor)
    else:
        xTensor = torch.Tensor(xArr)
        if createLabel:
            yTensor = torch.Tensor(yArr).type(torch.long)
    # xTensor is of shape: [grid id, seq]
    # yTensor is of shape: [grid id]
    if not createLabel:
        yTensor = []
    return xTensor, yTensor


def calc_netOutput_fromDataloader(data_in, my_net):
    dataset_uber = DataSet_oneLSTM_allGrid(data_in, my_net.sequence_size)
    dataloader_uber = data.DataLoader(dataset=dataset_uber, batch_size=my_net.batch_size, shuffle=False)
    net_accuracy = []
    net_rmse = []
    net_output = {0: []}
    label_output = {0: []}
    for inputs, labels in dataloader_uber:
        # compute test result of model
        localBatchSize = labels.shape[0]
        grid_size = labels.shape[1]
        testOut = my_net.forward(inputs)
        testOut = testOut.view(localBatchSize, my_net.class_size, grid_size)
        _, net_labels = torch.max(torch.exp(testOut.data), 1)
        net_labelsNp = net_labels.long().detach().numpy()
        labelsNp = labels.detach().numpy()
        testCorr = torch.sum(net_labels.long() == labels).detach().numpy()
        testTot = labels.size(0) * labels.size(1)
        # print("labTest:"+str(labTestNp.size)+", lables:"+str(labelsNp.size))
        rmse = np.sqrt(metrics.mean_squared_error(net_labelsNp.reshape(-1), labelsNp.reshape(-1)))
        net_accuracy.append(100 * testCorr / testTot)
        net_rmse.append(rmse)
        net_output[0].append(net_labelsNp)
        label_output[0].append(labelsNp)
    return net_output, label_output


def calc_netOutput_fromData(data_in, my_net, timeIndexs):
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

    net_output      = np.zeros((time_size, x_size, y_size))
    label_output    = np.zeros((time_size, x_size, y_size))
    # testOutTemp = []
    # netlabelTemp = []
    # inputTemp    = []
    for i, t in enumerate(timeIndexs):
        print("t:" + str(i) + " out of:" + str(timeIndexs.size))
        for x in range(x_size):
            for y in range(y_size):
                # print("t:" + str(i) + ",x:"+str(x)+", y:"+str(y))
                input, label = create_net_input(data_in[:, x, y].reshape([-1, 1, 1]), t, my_net.sequence_size)
                # input size is: [1, grid_id, seq_len]
                # label size is: [1, grid_id]
                testOut = my_net.forward(input)
                testOut = testOut.view(1, my_net.class_size, 1)
                _, net_labels = torch.max(torch.exp(testOut.data), 1)
                net_labelsNp = net_labels.long().detach().numpy()
                # testOutTemp.append(testOut.detach().numpy())
                # netlabelTemp.append(net_labelsNp)
                # inputTemp.append(input)
                labelsNp = label.detach().numpy()
                sizeMat = net_labelsNp.size
                net_rmse.append(np.sqrt(metrics.mean_squared_error(labelsNp.reshape(-1), net_labelsNp.reshape(-1))))
                net_accuracy.append(np.sum(labelsNp == net_labelsNp) / sizeMat)
                sizeMat_zeros = net_labelsNp[labelsNp == 0].size
                sizeMat_non_zeros = net_labelsNp[labelsNp != 0].size
                if (sizeMat_non_zeros > 0):
                    correct_non_zeros.append(
                        np.sum(net_labelsNp[labelsNp != 0] == labelsNp[labelsNp != 0]) / sizeMat_non_zeros)
                if sizeMat_zeros > 0:
                    correct_zeros.append(np.sum(net_labelsNp[labelsNp == 0] == labelsNp[labelsNp == 0]) / sizeMat_zeros)
                numEventsCreated.append(np.sum(labelsNp))
                numEventsPredicted.append(np.sum(net_labelsNp))
                net_output[i, x, y] = net_labelsNp
                label_output[i, x, y] = labelsNp


    # for i, t in enumerate(timeIndexs):
    #     input, label = create_net_input_allGrid(data_in, t, my_net.sequence_size)
    #     # input size is: [1, grid_id, seq_len]
    #     # label size is: [1, grid_id]
    #     testOut         = my_net.forward(input)
    #     testOut         = testOut.view(grid_size, my_net.class_size, 1)
    #     _, net_labels   = torch.max(torch.exp(testOut.data), 1)
    #     net_labelsNp    = net_labels.long().detach().numpy()
    #     labelsNp        = label.detach().numpy()
    #     sizeMat         = net_labelsNp.size
    #     net_rmse.append(np.sqrt(metrics.mean_squared_error(labelsNp.reshape(-1), net_labelsNp.reshape(-1))))
    #     net_accuracy.append(np.sum(labelsNp == net_labelsNp) / sizeMat)
    #     sizeMat_zeros       = net_labelsNp[labelsNp == 0].size
    #     sizeMat_non_zeros   = net_labelsNp[labelsNp != 0].size
    #     if (sizeMat_non_zeros > 0):
    #         correct_non_zeros.append(
    #             np.sum(net_labelsNp[labelsNp != 0] == labelsNp[labelsNp != 0]) / sizeMat_non_zeros)
    #     if sizeMat_zeros > 0:
    #         correct_zeros.append(
    #             np.sum(net_labelsNp[labelsNp == 0] == labelsNp[labelsNp == 0]) / sizeMat_zeros)
    #     numEventsCreated.append(np.sum(labelsNp))
    #     numEventsPredicted.append(np.sum(net_labelsNp))
    #     k = 0
    #     # save the output to matrix for each grid id -
    #     for x in range(x_size):
    #         for y in range(y_size):
    #             net_output[i, x, y] = net_labelsNp[k]
    #             label_output[i, x, y] = labelsNp[k]
    #             k += 1
    results = {'accuracy'           : net_accuracy,
               'rmse'               : net_rmse,
               'numCreated'         : numEventsCreated,
               'numPredicted'       : numEventsPredicted,
               'correctedZeros'     : correct_zeros,
               'correctedNonZeros'  : correct_non_zeros}
    return net_output, label_output, results


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
    epochs   = 'last'
    net_name     = 'smooth_20_seq_50_bs_40_hs_128_lr_0.5_ot_1_wd_0.002_torch.pkl'
    net_path     = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/'
    data_path    = '/Users/chanaross/dev/Thesis/UberData/'
    data_name    = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'
    ####################################################################################################################
    # this is for checking the actual output from the network during training (need to save results during training)
    # netName     = 'netDict'
    # labelsName  = 'labelsDict'
    # data_net    = loadFile(net_path + netName)
    # data_labels = loadFile(net_path + labelsName)
    # checkNetwork(data_net, data_labels, 'last')
    ####################################################################################################################

    network_path      = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/singleGridId_multiClassSmooth/'  # GPU_results/singleGridId_multiClassSmooth/'
    network_names     = ['smooth_10_seq_5_bs_40_hs_128_lr_0.05_ot_1_wd_0.002_torch.pkl']
    # network_names   = [f for f in os.listdir(network_path) if (f.endswith('.pkl'))]

    plot_graph_vs_time = True
    plot_time_gif      = False
    plot_loss_accuracy = True

    # create dictionary for storing result for each network tested
    results = {'networkName'            : [],
               'mean RMSE'              : [],
               'mean accuracy'          : [],
               'mean zeros accuracy'    : [],
               'mean nonZeros accuracy' : [],
               'mean smooth RMSE'               : [],
               'mean smooth accuracy'           : [],
               'mean smooth zeros accuracy'     : [],
               'mean smooth nonZeros accuracy'  : []}

    dataInputReal = np.load(data_path + data_name)
    # dataInputReal[dataInputReal > 1] = 1
    # use only data wanted (x, y, time)
    # xmin = 0    # 5
    # xmax = dataInputReal.shape[0]  # 6
    # ymin = 0   # 10
    # ymax = dataInputReal.shape[1]  # 11
    xmin = 5
    xmax = 6
    ymin = 10
    ymax = 11
    zmin = 48  # np.floor(dataInputReal.shape[2]*0.7).astype(int)
    zmax = dataInputReal.shape[2]
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:zmax]  # shrink matrix size for fast training in order to test model
    dataInputReal_orig = dataInputReal
    # dataInputReal = dataInputReal[5:6, 10:11, :]
    # reshape input data for network format -
    lengthT = dataInputReal.shape[2]
    lengthX = dataInputReal.shape[0]
    lengthY = dataInputReal.shape[1]
    dataInputReal = np.swapaxes(dataInputReal, 0, 1)
    dataInputReal = np.swapaxes(dataInputReal, 0, 2)
    # create results index's -
    tmin = 1100
    tmax = 4000
    timeIndexs = np.arange(tmin, tmax, 1).astype(int)
    xIndexs    = np.arange(xmin, xmax, 1).astype(int)
    yIndexs    = np.arange(ymin, ymax, 1).astype(int)
    numRuns = timeIndexs.size
    for network_name in network_names:
        print("network:" + network_name.replace('.pkl', ''))
        figPath     = network_path + 'figures/'
        fileName    = str(numRuns) + network_name.replace('.pkl', '')
        if "descent" in network_name:
            my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
        else:
            my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)
        my_net.eval()
        if plot_loss_accuracy:
            plotEpochGraphs(my_net, figPath, fileName)
        try:
            smoothParam = my_net.smoothingParam
        except:
            print("no smoothing param, using default 15")
            smoothParam = 15
        # create smooth data -
        dataInputSmooth      = moving_average(dataInputReal,      smoothParam, axis=0)     # smoothing data so that results are more clear to network
        dataInputSmooth_orig = moving_average(dataInputReal_orig, smoothParam, axis=2)

        # calc network output
        # real data -
        net_output_fromData, label_output_fromData, net_results = calc_netOutput_fromData(dataInputReal, my_net, timeIndexs)
        # smooth data -
        net_output_fromSmoothData, label_output_fromSmoothData, net_resultsSmooth = calc_netOutput_fromData(dataInputSmooth, my_net, timeIndexs)

        # plot each grid point vs. time
        if plot_graph_vs_time:
            # real data-
            checkNetwork(net_output_fromData, label_output_fromData, dataInputReal[timeIndexs, :, :],
                         timeIndexs, xIndexs, yIndexs, figPath, fileName, 'Real')
            # smooth data-
            checkNetwork(net_output_fromSmoothData, label_output_fromSmoothData, dataInputSmooth[timeIndexs, :, :],
                         timeIndexs, xIndexs, yIndexs, figPath, fileName, 'Smooth')

        # save total results to dictionary for each network tested -
        # real data -
        results['networkName'].append(network_name.replace('.pkl', ''))
        results['mean RMSE'].append(np.mean(net_results['rmse']))
        results['mean accuracy'].append(100 * np.mean(net_results['accuracy']))
        results['mean nonZeros accuracy'].append(100 * np.mean(net_results['correctedNonZeros']))
        results['mean zeros accuracy'].append(100 * np.mean(net_results['correctedZeros']))
        # smooth data -
        results['mean smooth RMSE'].append(np.mean(net_resultsSmooth['rmse']))
        results['mean smooth accuracy'].append(100 * np.mean(net_resultsSmooth['accuracy']))
        results['mean smooth nonZeros accuracy'].append(100 * np.mean(net_resultsSmooth['correctedNonZeros']))
        results['mean smooth zeros accuracy'].append(100 * np.mean(net_resultsSmooth['correctedZeros']))

        if plot_time_gif:
            # create gif for results -
            # all data -
            createTimeGif_allData(net_output_fromData, net_output_fromSmoothData, dataInputSmooth[timeIndexs, :, :], dataInputReal[timeIndexs, :, :], timeIndexs, fileName, figPath, 'Smooth')
            # # smooth data
            # createTimeGif(net_output_fromSmoothData, label_output_fromSmoothData, dataInputSmooth[timeIndexs, :, :], timeIndexs, fileName, figPath, 'Smooth')
            # # real data
            # createTimeGif(net_output_fromData, label_output_fromData, dataInputReal[timeIndexs, :, :], timeIndexs, fileName, figPath, 'Real')

        # print all results -
        # real data -
        print("average RMSE for " + str(numRuns) + " runs is:" + str(np.mean(net_results['rmse'])))
        print("average accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['accuracy'])))
        print("average corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['correctedZeros'])))
        print("average corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_results['correctedNonZeros'])))
        # smooth data -
        print("average smooth RMSE for " + str(numRuns) + " runs is:" + str(np.mean(net_resultsSmooth['rmse'])))
        print("average smooth accuracy for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_resultsSmooth['accuracy'])))
        print("average smooth corrected zeros " + str(numRuns) + " runs is:" + str(100 * np.mean(net_resultsSmooth['correctedZeros'])))
        print("average smooth corrected non zeros for " + str(numRuns) + " runs is:" + str(100 * np.mean(net_resultsSmooth['correctedNonZeros'])))

    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(network_path + 'FullGrid_results.csv')
    return




if __name__ == '__main__':
    main()
    print('Done.')

