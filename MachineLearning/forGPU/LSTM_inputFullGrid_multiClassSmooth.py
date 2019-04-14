import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
from matplotlib import pyplot as plt
import time, pickle, itertools
from sklearn import metrics
from math import sqrt
import sys
if torch.cuda.is_available():
    sys.path.insert(0, '/home/schanaby@st.technion.ac.il/thesisML/')
else:
    sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/')
from dataLoader_uber import DataSet_oneLSTM_allGrid

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
        print("turned x to gpu")
    return Variable(x)

isServerRun = torch.cuda.is_available()
if isServerRun:
    print ('Running using cuda')

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# creating optimization parameters and function
# adam    -(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
# SGD     -(params, lr=1e-3,momentum=0, dampening=0, weight_decay=0, nesterov=False)
# Adagrad -(params, lr=0.01, lr_decay=0, weight_decay=0)

def CreateOptimizer(netParams, ot, lr, dmp, mm, eps, wd):
    if ot == 1:
        optim = torch.optim.SGD(netParams, lr, mm, dmp, weight_decay=wd)
    elif ot == 2:
        optim = torch.optim.Adam(netParams, lr, (0.9, 0.999), eps, weight_decay=wd)
    elif ot == 3:
        optim = torch.optim.Adagrad(netParams, lr, weight_decay=wd)
    return optim


class Model(nn.Module):
    def __init__(self, grid_size, hidden_size, batch_size, sequence_size, class_size):
        super(Model, self).__init__()
        self.sequence_size    = sequence_size
        self.batch_size       = batch_size
        self.hiddenSize       = hidden_size
        self.gridSize         = grid_size
        self.class_size       = class_size

        self.loss           = None
        self.lossCrit       = None
        self.optimizer      = None
        self.lr             = None
        self.wd             = None
        self.lossWeights    = None
        self.maxEpochs      = None
        self.smoothingParam = None

        # output variables (loss, acc ect.)
        self.finalAcc       = 0
        self.finalLoss      = 0
        self.lossVecTrain   = []
        self.lossVecTest    = []
        self.accVecTrain    = []
        self.accVecTest     = []
        self.rmseVecTrain   = []
        self.rmseVecTest    = []

        self.lstm           = None
        self.fc_after_lstm  = None
        self.logSoftMax     = nn.LogSoftmax(dim=2)
        self.sigmoid        = nn.Sigmoid()

    def create_lstm(self, input_size):
        layer = nn.LSTM(input_size, self.hiddenSize, num_layers=2)
        return layer

    def create_fc_after_lstm(self, input_size, output_size):
        layer = nn.Sequential(nn.Linear(input_size, output_size))
        return layer

    def forward(self, x):
        batch_size = x.size(0)
        # x is of size : [batch_size , sequence_size, grid_size]
        # and needs to be seq then batch therefore view is done before entering lstm
        output, (h_n, c_n) = self.lstm(x.contiguous().view(self.sequence_size, batch_size, -1))
        out = self.fc_after_lstm(h_n[-1, :, :])
        out = out.view(batch_size, self.gridSize, self.class_size)
        out = self.logSoftMax(out)  # after last fc out is of size: [batch_size , grid_size, num_classes]
        # out = self.sigmoid(out.view(batch_size, -1))  # after last fc out is of size: [batch_size , num_classes] and is after LogSoftMax
        return out

    def calcLoss(self, outputs, labels):
        self.loss = self.lossCrit(outputs, labels)
        # if self.loss is None:
        #     self.loss = self.lossCrit(outputs, labels)
        # else:
        #     self.loss += self.lossCrit(outputs, labels).data

    # creating backward propagation - calculating loss function result
    def backward(self):
        self.loss.backward()

        # testing network on given test set

    def test_spesific(self, testLoader):
        # put model in evaluate mode
        self.eval()
        testCorr = 0.0
        testTot = 0.0
        localLossTest = []
        localAccTest = []
        localRmseTest = []
        for inputs, labels in testLoader:
            self.loss = None
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputVar = Variable(inputs)
            labVar = Variable(labels)
            # compute test result of model
            localBatchSize = labels.shape[0]
            grid_size      = labels.shape[1]
            testOut = self.forward(inputVar)
            testOut = testOut.view(localBatchSize, self.class_size, grid_size)
            _, labTest = torch.max(torch.exp(testOut.data), 1)
            self.calcLoss(testOut, labVar)
            localLossTest.append(self.loss.item())
            if isServerRun:
                labTestNp = labTest.type(torch.cuda.LongTensor).cpu().detach().numpy()
                labelsNp = labels.cpu().detach().numpy()
                testCorr = torch.sum(labTest.type(torch.cuda.LongTensor) == labels).cpu().detach().numpy() + testCorr
            else:
                labTestNp = labTest.long().detach().numpy()
                labelsNp = labels.detach().numpy()
                testCorr = torch.sum(labTest.long() == labels).detach().numpy() + testCorr
            testTot = labels.size(0) * labels.size(1) + testTot
            # print("labTest:"+str(labTestNp.size)+", lables:"+str(labelsNp.size))
            rmse = sqrt(metrics.mean_squared_error(labTestNp.reshape(-1), labelsNp.reshape(-1)))
            localAccTest.append(100 * testCorr / testTot)
            localRmseTest.append(rmse)
        accTest = np.average(localAccTest)
        lossTest = np.average(localLossTest)
        rmseTest = np.average(localRmseTest)
        print("test accuarcy is: {0}, rmse is: {1}, loss is: {2}".format(accTest, rmseTest, lossTest))
        return accTest, lossTest, rmseTest

        # save network

    def saveModel(self, path):
        torch.save(self, path)

def moving_average(data_set, periods=3, axis = 2):
    cumsum = np.cumsum(data_set, axis = axis)
    averageRes =  (cumsum[:, :, periods:] - cumsum[:, :, :-periods]) / float(periods)
    return np.floor(averageRes)

    # weights = np.ones(periods) / periods
    # convRes = np.convolve(data_set, weights, mode='valid')
    # return np.floor(convRes)


def main():
    #####################
    # Generate data
    #####################
    # data loader -
    if isServerRun:
        path = '/home/schanaby@st.technion.ac.il/thesisML/'
    else:
        path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName = '3D_allDataLatLonCorrected_20MultiClass_500gridpickle_30min.p'
    dataInput = np.load(path + fileName)

    flag_save_network = True

    xmin = 0
    xmax = dataInput.shape[0]
    ymin = 0
    ymax = dataInput.shape[1]
    zmin = 48
    dataInput     = dataInput[xmin:xmax, ymin:ymax, zmin:]  # shrink matrix size for fast training in order to test model
    dataInput     = dataInput[5:6, 10:11, :]
    smoothParam   = [10, 20, 30, 40]  #[10, 15, 30]

    testSize            = 0.2
    # define hyper parameters -
    hidden_sizeVec      = [128]  # [20, 64, 256, 512] #[20, 64, 264, 512]  # [20, 40, 64, 128]
    sequence_sizeVec    = [5, 10, 20]  # [5, 20, 30, 40]  # [5, 10, 15]  # length of sequence for lstm network
    batch_sizeVec       = [40]
    num_epochs          = 100

    # optimizer parameters -
    lrVec   = [0.05, 0.2, 0.5]  # [0.1, 0.5, 0.9] #[0.1, 0.5, 0.9]  # [0.1, 0.01, 0.001]
    otVec   = [1]  # [1, 2]
    dmp     = 0
    mm      = 0.9
    eps     = 1e-08
    wdVec   = [2e-3]

    # create case vectors
    networksDict = {}
    itr = itertools.product(smoothParam, sequence_sizeVec, batch_sizeVec, hidden_sizeVec, lrVec, otVec, wdVec)
    for i in itr:
        networkStr = 'smooth_{0}_seq_{1}_bs_{2}_hs_{3}_lr_{4}_ot_{5}_wd_{6}'.format(i[0], i[1], i[2], i[3], i[4], i[5], i[6])
        networksDict[networkStr] = {'seq': i[1], 'bs': i[2], 'hs': i[3], 'lr': i[4], 'ot': i[5], 'wd': i[6], 'sm': i[0]}

    for netConfig in networksDict:
        dataInputSmooth = moving_average(dataInput, networksDict[netConfig]['sm'])  # smoothing data so that results are more clear to network

        # dataInput[dataInput>1] = 1  # limit all events larger than 10 to be 10
        # define important sizes for network -
        x_size              = dataInputSmooth.shape[0]
        y_size              = dataInputSmooth.shape[1]
        dataSize            = dataInputSmooth.shape[2]
        class_size          = (np.max(np.unique(dataInputSmooth)) + 1).astype(int)

        num_train = int((1 - testSize) * dataSize)
        grid_size = x_size * y_size

        # output file
        outFile = open('LSTM_networksOutput.csv', 'w')
        outFile.write('Name;finalAcc;finalLoss;trainTime;numWeights;NumEpochs\n')


        print('Net Parameters: ' + netConfig)

        # create network based on input parameter's -
        hidden_size     = networksDict[netConfig]['hs']
        batch_size      = networksDict[netConfig]['bs']
        sequence_size   = networksDict[netConfig]['seq']
        lr              = networksDict[netConfig]['lr']
        ot              = networksDict[netConfig]['ot']
        wd              = networksDict[netConfig]['wd']

        my_net          = Model(grid_size, hidden_size, batch_size, sequence_size, class_size)
        my_net.lstm     = my_net.create_lstm(grid_size)  # lstm receives all grid points and seq length of
        my_net.fc_after_lstm = my_net.create_fc_after_lstm(my_net.hiddenSize, grid_size*class_size)
        my_net.to(device)
        print("model device is:")
        print(next(my_net.parameters()).device)
        numWeights = sum(param.numel() for param in my_net.parameters())
        print('number of parameters: ', numWeights)
        my_net.optimizer    = CreateOptimizer(my_net.parameters(), ot, lr, dmp, mm, eps, wd)
        my_net.lossCrit     = nn.NLLLoss(size_average=True)  # nn.BCELoss(size_average=True)

        my_net.maxEpochs = num_epochs
        my_net.lr        = lr
        my_net.wd        = wd
        my_net.smoothingParam = networksDict[netConfig]['sm']

        # network_path = '/Users/chanaross/dev/Thesis/MachineLearning/forGPU/GPU_results/limitedZero_500grid/'
        # network_name = 'gridSize11_epoch4_batch5_torch.pkl'
        # my_net = torch.load(network_path + network_name, map_location=lambda storage, loc: storage)

        # load data from data loader and create train and test sets
        data_train = dataInputSmooth[:, :, 0:num_train]
        data_test  = dataInputSmooth[:, :, num_train:]

        dataset_uber_train = DataSet_oneLSTM_allGrid(data_train, sequence_size)
        dataset_uber_test  = DataSet_oneLSTM_allGrid(data_test , sequence_size)

        # creating data loader
        dataloader_uber_train = data.DataLoader(dataset=dataset_uber_train, batch_size=batch_size, shuffle=False)
        dataloader_uber_test  = data.DataLoader(dataset=dataset_uber_test , batch_size=batch_size, shuffle=False)

        for numEpoch in range(num_epochs):
            my_net.loss = None
            # for each epoch, calculate loss for each batch -
            my_net.train()
            localLoss = [4]
            accTrain = [0]
            rmseTrain = [1]
            trainCorr = 0.0
            trainTot = 0.0
            if (1+numEpoch)%20 == 0:
                if my_net.optimizer.param_groups[0]['lr'] > 0.001:
                    my_net.optimizer.param_groups[0]['lr'] = my_net.optimizer.param_groups[0]['lr']/2
                else:
                    my_net.optimizer.param_groups[0]['lr'] = 0.001
            print('lr is: %.6f' % my_net.optimizer.param_groups[0]['lr'])
            for i, (input, labels) in enumerate(dataloader_uber_train):
                inputD = input.to(device)
                labelsD = labels.to(device)
                my_net.loss = None
                # create torch variables
                # input is of size [batch_size, grid_id, seq_size]
                inputVar = Variable(inputD).to(device)
                labVar   = Variable(labelsD).to(device)
                # if isServerRun:
                #     labVar   = labVar.type(torch.cuda.FloatTensor)
                # else:
                #     labVar   = labVar.type(torch.FloatTensor)
                # reset gradient
                my_net.optimizer.zero_grad()
                # forward
                grid_size        = labels.shape[1]
                local_batch_size = input.shape[0]
                # input to LSTM is [seq_size, batch_size, grid_size] , will be transferred as part of the forward
                netOut = my_net.forward(inputVar)
                netOut = netOut.view(local_batch_size, class_size, grid_size)
                _, labTrain = torch.max(torch.exp(netOut.data), 1)
                my_net.calcLoss(netOut, labVar)
                # backwards
                my_net.backward()
                # optimizer step
                my_net.optimizer.step()
                # local loss function list
                localLoss.append(my_net.loss.item())
                # if isServerRun:
                #     labTrain = labTrain.cpu()
                if isServerRun:
                    labTrainNp = labTrain.type(torch.cuda.LongTensor).cpu().detach().numpy()
                    # print("number of net labels different from 0 is:" + str(np.sum(labTrainNp > 0)))
                    # print("number of net labels 0 is:"+str(np.sum(labTrainNp == 0)))
                    labelsNp = labels.cpu().detach().numpy()
                    # print("number of real labels different from 0 is:" + str(np.sum(labelsNp > 0)))
                    # print("number of real labels 0 is:" + str(np.sum(labelsNp == 0)))
                    trainCorr = torch.sum(labTrain.type(torch.cuda.LongTensor) == labels).cpu().detach().numpy() + trainCorr
                else:
                    labTrainNp = labTrain.long().detach().numpy()
                    labelsNp = labels.detach().numpy()
                    trainCorr = torch.sum(labTrain.long() == labels).detach().numpy() + trainCorr
                trainTot = labels.size(0) * labels.size(1) + trainTot
                rmse = sqrt(metrics.mean_squared_error(labTrainNp.reshape(-1), labelsNp.reshape(-1)))
                accTrain.append(100 * trainCorr / trainTot)
                rmseTrain.append(rmse)
                # output current state
                if (i + 1) % 2 == 0:
                    print('Epoch: [%d/%d1 ], Step: [%d/%d], Loss: %.4f, Acc: %.4f, RMSE: %.4f'
                          % (numEpoch + 1, my_net.maxEpochs, i + 1,
                            dataloader_uber_train.batch_size,
                             my_net.loss.item(), accTrain[-1], rmseTrain[-1]))
                    # if (i+1) % 20 == 0:
                    #     if ((localLoss[-1] < np.max(np.array(localLoss[0:-1]))) or (accTrain[-1] > np.max(np.array(accTrain[0:-1])))) and flag_save_network:
                    #         # pickle.dump(my_net, open("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch+1) + "_batch" + str(i+1) + ".pkl", 'wb'))
                    #         my_net.saveModel("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch+1) + "_batch" + str(i+1) + "_torch.pkl")
                    #         # networkStr = "gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch+1) + "_batch" + str(i+1)
                    #         # outArray = np.stack([np.array(localLoss), np.array(accTrain)])
                    #         # np.save(networkStr + "_oArrBatch.npy", outArray)
            my_net.lossVecTrain.append(np.average(localLoss))
            my_net.accVecTrain.append(np.average(accTrain))
            my_net.rmseVecTrain.append(np.average(rmseTrain))
            # test network for each epoch stage
            accEpochTest, lossEpochTest, rmseEpochTest = my_net.test_spesific(testLoader=dataloader_uber_test)
            my_net.accVecTest.append(accEpochTest)
            my_net.lossVecTest.append(lossEpochTest)
            my_net.rmseVecTest.append(rmseEpochTest)
            if (flag_save_network):
                my_net.saveModel(netConfig + "_torch.pkl")
                # outArray = np.stack([np.array(my_net.lossVecTest), np.array(my_net.lossVecTrain),
                #                      np.array(my_net.accVecTest), np.array(my_net.accVecTrain)])
                # np.save("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch)  + "_oArrBatch.npy", outArray)
        my_net.finalAcc  = accEpochTest
        my_net.finalLoss = lossEpochTest
        my_net.finalRmse = rmseEpochTest
        # name, HyperPerams, accur, num total weights
        # err vs epoch, loss vs epoch,
        strWrite = '{0};{1};{2};{3};{4}\n'.format(netConfig, my_net.finalAcc, my_net.finalLoss, numWeights, my_net.maxEpochs)
        outFile.write(strWrite)

    outFile.close()

    return


if __name__ == '__main__':
    main()
    print('Done.')
