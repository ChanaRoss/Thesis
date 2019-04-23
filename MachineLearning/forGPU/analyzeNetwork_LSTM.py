# mathematical imports
import numpy as np
from sklearn import metrics
# graphical imports
from matplotlib import pyplot as plt
import seaborn as sns
# system imports
import pickle
import sys
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


def loadFile(fileName):
    with open(fileName+'.pkl', 'rb') as handle:
        data = pickle.load(handle)
    return data

def checkNetwork(data_net, data_labels, epochs = 'last'):

    clr = np.random.rand(3, 9999)
    lastTime = 0
    for i in range(len(data_labels[0])):
        timeIndex = np.array(list(range(len(data_labels[0][i]))))
        plt.plot(timeIndex + lastTime, data_labels[0][i], label='batch num:' + str(i), color=clr[:, i], marker='*')
        lastTime = timeIndex.max() + lastTime+1

    lastTime = 0
    if epochs == 'last':  # only plot last epoch results
        data = data_net[len(data_net) - 1]
        for b in range(len(data)):
            timeIndex = np.array(list(range(len(data[b]))))
            plt.plot(timeIndex + lastTime, data[b], label='net, batch num:' + str(b), linestyle='-.', color=clr[:, b],
                     marker='o')
            lastTime = timeIndex.max() + lastTime+1
    else:  # plot all epoch results
        for i, data in data_net.items():
            lastTime = 0
            for b in range(len(data)):
                timeIndex = np.array(list(range(len(data[b]))))
                plt.plot(timeIndex+lastTime, data[b], label='net, batch num:'+str(b), linestyle='-.', color = clr[:, b])
                lastTime = timeIndex.max()+lastTime+1
    #plt.show()


def create_net_input(data, t_index, seq_len):
    t_size = data.shape[0]
    x_size = data.shape[1]
    y_size = data.shape[2]
    temp = np.zeros(shape=(seq_len, x_size, y_size))
    if (t_index - seq_len > 0):
        temp = data[t_index - seq_len:t_index, :, :]
    else:
        temp[seq_len - t_index:, :, :] = data[0:t_index, :, :]
    xArr = np.zeros(shape=(1, x_size * y_size, seq_len))
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

    yArr = np.zeros(shape=(1, x_size* y_size))
    k = 0
    for i in range(x_size):
        for j in range(y_size):
            xArr[0, k, :] = temp[:, i, j]
            yArr[0, k] = yArrTemp[i, j]
            k += 1

    if torch.cuda.is_available():
        xTensor = torch.Tensor(xArr).cuda()
        yTensor = torch.Tensor(yArr).type(torch.cuda.LongTensor)
    else:
        xTensor = torch.Tensor(xArr)
        yTensor = torch.Tensor(yArr).type(torch.long)
    # xTensor is of shape: [grid id, seq]
    # yTensor is of shape: [grid id]
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


def calc_netOutput_fromData(data_in, my_net):
    time_indexs = np.array(range(data_in.shape[0]))
    net_accuracy = []
    net_rmse = []
    net_output = {0: []}
    label_output = {0: []}
    for t in time_indexs:
        input, label = create_net_input(data_in, t, my_net.sequence_size)
        grid_size = label.shape[1]
        testOut = my_net.forward(input)
        testOut = testOut.view(1, my_net.class_size, grid_size)
        _, net_labels = torch.max(torch.exp(testOut.data), 1)
        net_labelsNp = net_labels.long().detach().numpy()
        labelsNp = label.detach().numpy()
        testCorr = torch.sum(net_labels.long() == label).detach().numpy()
        testTot = label.size(0) * label.size(1)
        # print("labTest:"+str(labTestNp.size)+", lables:"+str(labelsNp.size))
        rmse = np.sqrt(metrics.mean_squared_error(net_labelsNp.reshape(-1), labelsNp.reshape(-1)))
        net_accuracy.append(100 * testCorr / testTot)
        net_rmse.append(rmse)
        net_output[0].append(net_labelsNp)
        label_output[0].append(labelsNp)
    return net_output, label_output




def main():
    epochs   = 'last'
    net_name     = 'smooth_20_seq_50_bs_40_hs_128_lr_0.5_ot_1_wd_0.002_torch.pkl'
    net_path     = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/'
    data_path    = '/Users/chanaross/dev/Thesis/UberData/'
    data_name    = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'

    netName     = 'netDict'
    labelsName  = 'labelsDict'
    data_net    = loadFile(net_path + netName)
    data_labels = loadFile(net_path + labelsName)

    checkNetwork(data_net, data_labels, 'last')
    my_net = torch.load(net_path + net_name, map_location=lambda storage, loc: storage)
    my_net.eval()

    dataInputReal = np.load(data_path + data_name)
    xmin = 5  # 0
    xmax = 6  # dataInputReal.shape[0]
    ymin = 10
    ymax = 11  # dataInputReal.shape[1]
    zmin = 48  # np.floor(dataInputReal.shape[2]*0.7).astype(int)
    dataInputReal = dataInputReal[xmin:xmax, ymin:ymax, zmin:]  # shrink matrix size for fast training in order to test model
    # dataInputReal = dataInputReal[5:6, 10:11, :]
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
    net_output, label_output = calc_netOutput_fromDataloader(dataInputSmooth, my_net)
    dataInputReal = np.swapaxes(dataInputReal, 0, 1)
    dataInputReal = np.swapaxes(dataInputReal, 0, 2)
    dataInputSmooth = np.swapaxes(dataInputSmooth, 0, 1)
    dataInputSmooth = np.swapaxes(dataInputSmooth, 0, 2)
    net_output_fromData, label_output_fromData = calc_netOutput_fromData(dataInputSmooth, my_net)
    plt.figure()
    timeReal = np.array(list(range(dataInputSmooth.shape[0])))
    plt.plot(timeReal, dataInputSmooth[:, 0, 0], color='k', label='real data')
    checkNetwork(net_output, label_output, 'last')
    plt.figure()
    plt.plot(timeReal, dataInputSmooth[:, 0, 0], color='k', label='real data')
    checkNetwork(net_output_fromData, label_output_fromData, 'last')
    plt.show()
    return




if __name__ == '__main__':
    main()
    print('Done.')

