import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/')
from CNN_LSTM_NeuralNetOrigV2 import Model


# matplotlib.use('Agg')  # allows it to operate on machines without graphical interface


def save_plots(results,FigName,fileName):
    # plt.clf()
    lgstr = []
    color_p = ['r', 'b', 'm', 'k', 'g', 'y']
    lineType = ['-', '-.']
    for i in range(0, len(results[0])):
        plt.plot(range(1, len(results[0][i])+1), results[0][i], color=color_p[i], linewidth=1.2, linestyle=lineType[0])
        plt.plot(range(2, len(results[1][i])+2), results[1][i], color=color_p[i+1], linewidth=1.2, linestyle=lineType[1])
        lgstr.append('Train - SGD optimizer')
        lgstr.append('Test - SGD optimizer')
        # lgstr.append('Train ' + str(i+1))
        # lgstr.append('Test ' + str(i+1))

    plt.title(FigName)
    strPlt = FigName.split('Vs')
    plt.ylabel(strPlt[0])
    plt.xlabel(strPlt[1])
    plt.grid(True)
    plt.legend(lgstr, loc='best')
    fileNameSave =  fileName.split('.pkl')
    plt.savefig(fileNameSave[0] + '_' + FigName + '.jpg')



path     = []
filename = []


path.append('/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/')
filename.append('/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/gridSize21_epoch95_batch4_torch.pkl')



for i in range(0,len(path)):
    my_net = torch.load(filename[i], map_location=lambda storage, loc: storage)
    data = np.stack([np.array(my_net.accVecTrain), np.array(my_net.lossVecTrain), np.array(my_net.rmseVecTrain),
                     np.array(my_net.accVecTest), np.array(my_net.lossVecTest), np.array(my_net.rmseVecTest)])
    epochs      = []
    acc_test    = []
    acc_train   = []
    loss_test   = []
    loss_train  = []
    rmse_train  = []
    rmse_test   = []
    epochs.append(range(len(data[0])))
    acc_train.append(data[0])
    loss_train.append(data[1])
    rmse_train.append(data[2])
    acc_test.append(data[3])
    loss_test.append(data[4])
    rmse_test.append(data[5])

to_plot = [acc_train, acc_test]
save_plots(to_plot, 'AccVsEpoch', 'Comparison')
plt.show()
to_plot = [loss_train, loss_test]
save_plots(to_plot, 'LossVsEpoch', 'Comparison')
plt.show()
to_plot = [rmse_train, rmse_test]
save_plots(to_plot, 'RmseVsEpoch', 'Comparison')
plt.show()