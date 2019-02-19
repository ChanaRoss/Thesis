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
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/')
from CNN_LSTM_NeuralNet import Model


# matplotlib.use('Agg')  # allows it to operate on machines without graphical interface


def save_plots(results,FigName,fileName):
    # plt.clf()
    lgstr = []
    color_p = ['r','b','m','k','g','y']
    lineType = ['-','-.']
    for i in range(0,len(results[0])):
        plt.plot(range(1,len(results[0][i])+1),results[0][i],color = color_p[i],linewidth = 1.2,linestyle = lineType[0])
        plt.plot(range(2,len(results[1][i])+2),results[1][i],color = color_p[i+1],linewidth = 1.2,linestyle = lineType[1])
        lgstr.append('Train - triple Augmentation , ADAM optimizer')
        lgstr.append('Test - triple Augmentation , ADAM optimizer')
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



path = []
filename = []


path.append('/Users/chanaross/dev/Thesis/MachineLearning/')
filename.append('gridSize20_epoch0_batch84.npy')

#
# my_net = torch.load(filename[-1])
#
# # my_net = torch.load(filename[-1], map_location=lambda storage, loc: storage)
#
# data = np.stack([np.array([]), np.array(my_net.lossVecTrain),
#                                      np.array([]), np.array(my_net.accVecTrain)])
epochs = []
acc_test = []
acc_train = []
loss_test = []
loss_train = []
for i in range(0,len(path)):
    data = np.load(path[i] + filename[i])
    epochs.append(range(len(data[0])))
    loss_test.append(data[0])
    loss_train.append(data[1])
    acc_test.append(data[2])
    acc_train.append(data[3])


to_plot = [acc_train,acc_test]
save_plots(to_plot,'AccVsEpoch','Comparison')
plt.show()
to_plot = [loss_train,loss_test]
save_plots(to_plot,'LossVsEpoch','Comparison')
plt.show()