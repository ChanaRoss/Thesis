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
sys.path.insert(0, '/home/schanaby@st.technion.ac.il/thesisML/')
from dataLoader_uber import DataSetCnn_LSTM_NonZero

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

def CreateOptimizer(netParams, ot, lr, dmp, mm, eps):
    if ot == 1:
        optim = torch.optim.SGD(netParams, lr, mm, dmp)
    elif ot == 2:
        optim = torch.optim.Adam(netParams, lr, (0.9, 0.999), eps)
    elif ot == 3:
        optim = torch.optim.Adagrad(netParams, lr)
    return optim


class Model(nn.Module):
    def __init__(self, cnn_input_size, class_size, hidden_size, batch_size, sequence_size, kernel_size,
                 stride_size, num_cnn_features, num_cnn_layers, fc_after_cnn_out_size, cnn_input_dimension):
        super(Model, self).__init__()
        self.sequence_size    = sequence_size
        self.hiddenSize       = hidden_size
        self.batch_size       = batch_size
        self.kernel_size      = kernel_size
        self.stride_size      = stride_size
        self.cnn_input_size   = cnn_input_size
        self.class_size       = class_size
        self.fc_output_size   = fc_after_cnn_out_size
        self.num_cnn_features = num_cnn_features
        self.num_cnn_layers   = num_cnn_layers
        self.cnn_input_dimension = cnn_input_dimension

        self.loss       = None
        self.lossCrit   = None
        self.optimizer  = None
        self.lr         = None
        self.maxEpochs  = None
        # output variables (loss, acc ect.)
        self.finalAcc       = 0
        self.finalLoss      = 0
        self.lossVecTrain   = []
        self.lossVecTest    = []
        self.accVecTrain    = []
        self.accVecTest     = []
        self.rmseVecTrain   = []
        self.rmseVecTest    = []

        self.cnn            = nn.ModuleList()
        self.fc_after_cnn   = nn.ModuleList()
        self.lstm           = None
        self.fc_after_lstm  = None
        self.logSoftMax     = nn.LogSoftmax(dim=1)


    def create_cnn(self):
        padding_size = int(0.5*(self.kernel_size - 1))
        # defines cnn network
        layers = []
        for i in range(self.num_cnn_layers):
            if i == 0:
                layers += [nn.Conv2d(self.cnn_input_size, self.num_cnn_features, kernel_size=self.kernel_size, stride=self.stride_size, padding=padding_size),
                           # nn.BatchNorm2d(self.num_cnn_features),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(self.num_cnn_features, self.num_cnn_features, kernel_size=self.kernel_size, stride=self.stride_size, padding=padding_size),
                           # nn.BatchNorm2d(self.num_cnn_features),
                           nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def create_lstm(self, input_size):
        layer = nn.LSTM(input_size, self.hiddenSize)
        return layer

    def create_fc_after_cnn(self, input_size, output_size):
        layer = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        return layer

    def create_fc_after_lstm(self, input_size, output_size):
        layer = nn.Sequential(nn.Linear(input_size, output_size))
        return layer


    def forward(self,x):
        batch_size = x.size(0)
        cnn_output = torch.zeros([batch_size, self.fc_output_size, self.sequence_size]).to(device)
        # x is of size : [batch_size , mat_x , mat_y , sequence_size]
        for i in range(self.sequence_size):
            xtemp = x[:, i, :, :].view(x.size(0), 1, x.size(2), x.size(3))
            out = self.cnn[i](xtemp)
            out = out.view((batch_size, -1))
            out = self.fc_after_cnn[i](out)  # after fully connected out is of size : [batch_size, fully_out_size]
            cnn_output[:, :, i] = out
        output, (h_n, c_n) = self.lstm(cnn_output.view(self.sequence_size, batch_size, -1))
        out = self.fc_after_lstm(h_n)
        out = self.logSoftMax(out.view(batch_size,-1))  # after last fc out is of size: [batch_size , num_classes] and is after LogSoftMax
        return out

    def calcLoss(self, outputs, labels):
        if self.loss is None:
            self.loss = self.lossCrit(outputs, labels)
        else:
            self.loss += self.lossCrit(outputs, labels).data
        # self.loss = self.lossCrit(outputs, labels)

    # creating backward propagation - calculating loss function result
    def backward(self):
        self.loss.backward(retain_graph=True)

        # testing network on given test set

    def test_spesific(self, testLoader):
        # put model in evaluate mode
        self.eval()
        testCorr = 0.0
        testTot = 0.0
        localLossTest = []
        localAccTest = []
        localRmseTest = []
        self.loss = None
        for inputs, labels in testLoader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputVar = Variable(inputs)
            labVar = Variable(labels)
            # compute test result of model
            localBatchSize = labels.shape[0]
            grid_size      = labels.shape[1]
            testOut = torch.zeros((localBatchSize, self.class_size, grid_size)).to(device)
            labTest = torch.zeros((localBatchSize, grid_size)).to(device)
            for k in range(grid_size):
                testOut = self.forward(inputVar[:, :, :, :, k])
                _, labTest[:, k] = torch.max(testOut.data, 1)
                self.calcLoss(testOut, labVar[:, k])
            # self.backward(testOut, labVar)
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
        print("test accuarcy is: {0}, rmse is: {1}".format(accTest, rmseTest))
        return accTest, lossTest, rmseTest

        # save network

    def saveModel(self, path):
        torch.save(self, path)




def main():
    #####################
    # Generate data
    #####################
    # data loader -
    if isServerRun:
        path = '/home/schanaby@st.technion.ac.il/thesisML/'
    else:
        path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_allData.p'
    dataInput = np.load(path + fileName)

    flag_save_network = True

    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 40
    # should only take from busy time in order to learn distribution
    zmin = 16000
    zmax = 24000  # 32000
    dataInput     = dataInput[xmin:xmax, ymin:ymax, zmin:zmax]  # shrink matrix size for fast training in order to test model
    # define important sizes for network -
    x_size              = dataInput.shape[0]
    y_size              = dataInput.shape[1]
    dataSize            = dataInput.shape[2]
    classNum            = (np.max(np.unique(dataInput)) + 1).astype(int)
    testSize            = 0.2
    sequence_size       = 5  # length of sequence for lstm network
    cnn_input_size      = 1  # size of matrix in input cnn layer  - each sequence goes into different cnn network
    cnn_dimension       = 9  # size of matrix around point i for cnn network
    batch_size          = 200
    num_epochs          = 100
    max_dataloader_size = 50
    num_train           = int((1 - testSize) * dataSize)
    # define hyper parameters -
    hidden_size         = 64
    kernel_size         = 3
    stride_size         = 1
    num_cnn_features    = 128
    num_cnn_layers      = 3
    fc_after_cnn_out_size = 64

    # optimizer parameters -
    lr  = 0.001
    ot  = 1
    dmp = 0
    mm  = 0.9
    eps = 1e-08


    # create network based on input parameter's -
    my_net = Model(cnn_input_size, classNum, hidden_size, batch_size, sequence_size, kernel_size,
                   stride_size, num_cnn_features, num_cnn_layers, fc_after_cnn_out_size, cnn_dimension)
    for i in range(sequence_size):
        my_net.cnn.append(my_net.create_cnn())
        my_net.fc_after_cnn.append(my_net.create_fc_after_cnn(num_cnn_features*cnn_dimension*cnn_dimension, fc_after_cnn_out_size))
    my_net.lstm = my_net.create_lstm(fc_after_cnn_out_size)
    my_net.fc_after_lstm = my_net.create_fc_after_lstm(my_net.hiddenSize, classNum)
    # # setup network
    # if isServerRun:
    #     my_net = my_net.cuda()
    #     print("model converted to cuda mode")
    my_net.to(device)
    print("model device is:")
    print(next(my_net.parameters()).device)
    numWeights = sum(param.numel() for param in my_net.parameters())
    print('number of parameters: ', numWeights)
    my_net.optimizer = CreateOptimizer(my_net.parameters(), ot, lr, dmp, mm, eps)
    loss_weights = np.ones(classNum)
    loss_weights[0] = 0.05
    w = torch.tensor(list(loss_weights), dtype=torch.float).to(device)
    my_net.lossCrit = nn.NLLLoss(weight=w)
    my_net.maxEpochs = num_epochs


    # load data from data loader and create train and test sets
    data_train = dataInput[:, :, 0:num_train]
    data_test  = dataInput[:, :, num_train:]

    dataset_uber_train = DataSetCnn_LSTM_NonZero(data_train, sequence_size, cnn_dimension, max_dataloader_size)
    dataset_uber_test  = DataSetCnn_LSTM_NonZero(data_test, sequence_size, cnn_dimension, max_dataloader_size)

    # creating data loader
    dataloader_uber_train = data.DataLoader(dataset=dataset_uber_train, batch_size=batch_size, shuffle=True)
    dataloader_uber_test  = data.DataLoader(dataset=dataset_uber_test, batch_size=batch_size, shuffle=False)

    for numEpoch in range(num_epochs):
        my_net.loss = None
        # for each epoch, calculate loss for each batch -
        my_net.train()
        localLoss = [4]
        accTrain = [0]
        rmseTrain = [1]
        trainCorr = 0.0
        trainTot = 0.0
        if (1+numEpoch)%10 == 0:
            if my_net.optimizer.param_groups[0]['lr']>0.0001:
                my_net.optimizer.param_groups[0]['lr'] = my_net.optimizer.param_groups[0]['lr']/2
            else:
                my_net.optimizer.param_groups[0]['lr'] = 0.001
        print('lr is: %.6f' % my_net.optimizer.param_groups[0]['lr'])
        for i, (input, labels) in enumerate(dataloader_uber_train):
            inputD = input.to(device)
            labelsD = labels.to(device)
            my_net.loss = None
            # create torch variables
            # input is of size [batch_size, seq_len, x_inputCnn, y_inputCnn, grid_id]
            inputVar = Variable(inputD).to(device)
            labVar = Variable(labelsD).to(device)
            # reset gradient
            my_net.optimizer.zero_grad()
            # forward
            labTrain = torch.tensor([]).to(device)
            labTrain = labTrain.new_zeros(labVar.size())
            grid_size = labels.shape[1]
            for k in range(grid_size):
                netOut = my_net.forward(inputVar[:, :, :, :, k])
                _, labTrain[:, k] = torch.max(torch.exp(netOut.data), 1)
                my_net.calcLoss(netOut, labVar[:, k])
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
                labelsNp = labels.cpu().detach().numpy()
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
            if (i + 1) % 5 == 0:
                print('Epoch: [%d/%d1 ], Step: [%d/%d], Loss: %.4f, Acc: %.4f, RMSE: %.4f'
                      % (numEpoch + 1, my_net.maxEpochs, i + 1,
                        dataloader_uber_train.batch_size,
                         my_net.loss.item(), accTrain[-1], rmseTrain[-1]))
                if ((localLoss[-1] > np.max(np.array(localLoss[0:-1]))) or (accTrain[-1] > np.max(np.array(accTrain[0:-1])))) and flag_save_network:
                    pickle.dump(my_net, open("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch) + "_batch" + str(i) + ".pkl", 'wb'))
                    my_net.saveModel("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch) + "_batch" + str(i) + "_torch.pkl")
                    networkStr = "gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch) + "_batch" + str(i)
                    outArray = np.stack([np.array(localLoss), np.array(accTrain)])
                    np.save(networkStr + "_oArrBatch.npy", outArray)
        my_net.lossVecTrain.append(np.average(localLoss))
        my_net.accVecTrain.append(np.average(accTrain))
        my_net.rmseVecTrain.append(np.average(rmseTrain))
        # test network for each epoch stage
        accEpochTest, lossEpochTest, rmseEpochTest = my_net.test_spesific(testLoader=dataloader_uber_test)
        my_net.accVecTest.append(accEpochTest)
        my_net.lossVecTest.append(lossEpochTest)
        my_net.rmseVecTest.append(rmseEpochTest)
        if (flag_save_network):
            outArray = np.stack([np.array(my_net.lossVecTest), np.array(my_net.lossVecTrain),
                                 np.array(my_net.accVecTest), np.array(my_net.accVecTrain)])
            np.save("gridSize" + str(xmax - xmin) + "_epoch" + str(numEpoch)  + "_oArrBatch.npy", outArray)
    my_net.finalAcc = accEpochTest
    my_net.finalLoss = lossEpochTest
    my_net.finalRmse = rmseEpochTest
    endTime = time.process_time()

    return


if __name__ == '__main__':
    main()
    print('Done.')
