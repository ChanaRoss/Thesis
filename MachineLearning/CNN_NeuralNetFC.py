import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # defining linear size and number of channels
        self.fcSize = None
        self.sizeX = None
        self.sizeY = None
        self.sizeClasses = None
        # defining convolution network sizes
        self.kernalSize = None
        self.paddingSize = None
        self.strideSize = None
        # defining pooling sizes
        self.poolingKernalSize = None
        self.poolingStrideSize = None
        # creating network architecture
        self.features = None
        self.classifier = None
        # create backprop fields
        self.optimizer = None
        self.logsoftmax = nn.LogSoftmax()
        self.lossCrit = None
        self.loss = None
        self.lr = None
        self.p_dp = None
        self.maxEpochs = None
        # output variables (loss, acc ect.)
        self.finalAcc = 0
        self.finalLoss = 10
        self.lossVecTrain = []
        self.lossVecTest = []
        self.accVecTrain = []
        self.accVecTest = []

    def makeLayers(self, inputParam, in_channels):
        layers = []
        for x in inputParam:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=self.poolingKernalSize, stride=self.poolingStrideSize)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernalSize, stride=self.strideSize, padding=self.paddingSize),
                           #nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        # layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        batchSize = out.size(0)
        out = out.view(batchSize, -1)
        out = self.classifier(out)
        # out dimension is: [batch size , num channels , height ,width ]
        out = out.view(batchSize*self.sizeX*self.sizeY, -1)
        out = self.logsoftmax(out)
        out = out.view(batchSize, self.sizeClasses, self.sizeX, self.sizeY)
        # out = out.view_as(outTemp)
        return out

    # creating backward propagation - calculating loss function result
    def backward(self, outputs, labels):
        self.loss = self.lossCrit(outputs, labels)
        self.loss.backward()

    # calculating fc layer size for linear layer
    def get_flat_fts(self,in_size,fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))

    # testing network on given test set
    def test_spesific(self,testLoader):

        # put model in evaluate mode
        self.eval()
        testCorr = 0.0
        testTot = 0.0
        localLossTest = []
        localAccTest = []
        for images, labels in testLoader:
            imgVar = to_var(images)
            labVar = to_var(labels)
            # compute test result of model
            testOut = self.forward(imgVar)
            # find loss of test set
            self.backward(testOut, labVar)
            localLossTest.append(self.loss.item())
            _, labTest = torch.max(testOut.data, 1)
            if torch.cuda.is_available():
                labTest = labTest.cpu()
            testCorr = torch.sum(labTest == labels).detach().numpy() + testCorr
            testTot = labels.size(0)*labels.size(1)*labels.size(2) + testTot
            localAccTest.append(100 * testCorr / testTot)
        accTest = np.average(localAccTest)
        lossTest = np.average(localLossTest)
        print("test accuarcy is: {0}".format(accTest))
        return accTest, lossTest

    # save network
    def saveModel(self, path):
        torch.save(self, path)
