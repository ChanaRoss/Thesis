import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
# import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
# my imports -
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/')
from CNN_NeuralNet import CNN
from dataLoader_uber import DataSetCnn

import numpy as np
import time, pickle, itertools



isServerRun = torch.cuda.is_available()
if isServerRun:
    print ('Running using cuda')


# creating cuda variable if running on GPU
def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
# creating optimization parameters and function
# adam    -(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0)
# SGD     -(params, lr=1e-3,momentum=0, dampening=0, weight_decay=0, nesterov=False)
# Adagrad -(params, lr=0.01, lr_decay=0, weight_decay=0)

def CreateOptimizer(netParams,optimParamDict):
    if optimParamDict['ot'] == 1:
        optim = torch.optim.SGD(netParams,optimParamDict['lr'],
                                optimParamDict['mm'],optimParamDict['dmp'],
                                optimParamDict['wd'],nesterov=False)
    elif optimParamDict['ot'] == 2:
        optim = torch.optim.Adam(netParams, optimParamDict['lr'],
                                 (0.9, 0.999), optimParamDict['eps'],
                                optimParamDict['wd'])
    elif optimParamDict['ot'] == 3:
        optim = torch.optim.Adagrad(netParams, optimParamDict['lr'],
                                optimParamDict['lrd'], optimParamDict['wd'])
    return optim

# data loader
path = '/home/chanaby/Documents/Thesis/machineLearning/'
fileName = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_wday_1.p'
dataInput = np.load(path + fileName)
xmin = 0
xmax = 5
ymin = 0
ymax = 5
dataInput   = dataInput[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model
numClasses  = np.unique(dataInput).size
dataSize    = dataInput.shape[2]
testSize    = 0.2
sequenceDim = 5
num_train   = int((1 - testSize) * dataSize)

data_train = dataInput[:, :, 0:num_train]
data_test = dataInput[:, :, num_train + 1:]

dataset_uber_train = DataSetCnn(data_train, sequenceDim)
dataset_uber_test  = DataSetCnn(data_test, sequenceDim)


# constant parameters
inX        = dataInput.shape[0]
inY        = dataInput.shape[1]
inChannels = sequenceDim
inputSize  = (inChannels, inX, inY)  # size of input in data
classNum   = np.unique(dataInput).size  # number of classes optional, number of simultaneous events per grid point
maxEpochs  = 50
# hyper parameters
batchSize = [50]#[50,100]
p_dp = [0.5]
# non-constant parameters
conv1Vec = [8,16,32]#[8,16,32,64]
conv2Vec = [16,32]#[16,32,64]
conv3Vec = [32,32]#[0,32,64]
conv4Vec = [32,64]#[0,32,64]
conv5Vec = [classNum]#[0,32,64]
# convolution parameters
kernalSize = [3]#[3,5]
strideSize = [1]
# max pooling parameters
poolingKernalSize = [2]
poolingStrideSize = [2]
# optimization parameters
optimType = [1]#[1,2]
lrVec = [0.01]
weightDecay = [0.0]
momentum = [0.9]
dampening = [0]
epsilon = [1e-8]
lrDecay = [0]
# based on convention from the literature
paddingSize = [int(0.5*(i-1)) for i in kernalSize]
# create case vectors
networksDict = {}
itr = itertools.product(batchSize,conv1Vec,conv2Vec,conv3Vec,kernalSize,strideSize,paddingSize,
                        poolingKernalSize,poolingStrideSize,
                        optimType,lrVec,weightDecay,momentum,dampening,conv5Vec,epsilon,lrDecay,p_dp,conv4Vec)
for i in itr:
    networkStr = 'bs_{0}_cnv1_{1}_cnv2_{2}_cnv3_{3}_ks_{4}_ss_{5}_ps_{6}' \
                 '_pks_{7}_pss_{8}_ot_{9}_lrvec_{10}_wd_{11}_mm_{12}_dmp_{13}_cnv5_{14}_eps_{15}_lrd_{16}_pdp_{17}_cnv4_{18}_cnv6_32_cnv7_32' \
        .format(i[0], i[1],i[2],i[3],i[4],i[5],i[6],i[7],i[8],i[9], i[10],i[11],i[12],i[13],i[14],i[15],i[16],i[17],i[18])
    networksDict[networkStr] = {'bs': i[0], 'cnv1': i[1], 'cnv2': i[2], 'cnv3': i[3], 'ks': i[4], 'ss': i[5],'ps': i[6],
                                'pks': i[7], 'pss': i[8], 'ot': i[9], 'lr': i[10], 'wd': i[11], 'mm': i[12], 'dmp' : i[13],
                                'cnv5' : i[14],'eps':i[15],'lrd':i[16],'p_dp':i[17],'cnv4':i[18]}

# output file
outFile = open('CNN_networksOutput.csv', 'w')
outFile.write('Name;finalAcc;finalLoss;trainTime;numWeights;NumEpochs\n')


for netConfig in networksDict:
    print('Net Parameters: ' + netConfig)
    Net = CNN()

    # creating data loader
    dataloader_uber_train = data.DataLoader(dataset=dataset_uber_train, batch_size=networksDict[netConfig]['bs'], shuffle=True)
    dataloader_uber_test  = data.DataLoader(dataset=dataset_uber_test, batch_size=networksDict[netConfig]['bs'], shuffle=False)

    # defining parameters for convolution layers
    Net.kernalSize = networksDict[netConfig]['ks']
    Net.paddingSize = networksDict[netConfig]['ps']
    Net.strideSize = networksDict[netConfig]['ss']
    Net.poolingKernalSize = networksDict[netConfig]['pks']
    Net.poolingStrideSize = networksDict[netConfig]['pss']
    # list of Convolution layout and size
    featursParam = [networksDict[netConfig]['cnv1'], networksDict[netConfig]['cnv2'],
                    networksDict[netConfig]['cnv3'], networksDict[netConfig]['cnv4'], networksDict[netConfig]['cnv5']]
    inputClassifier = networksDict[netConfig]['cnv3']
    # if networksDict[netConfig]['cnv4'] == 0: # only 2 convolution layers
    #     featursParam = [networksDict[netConfig]['cnv1'],'M', networksDict[netConfig]['cnv2'],'M',networksDict[netConfig]['cnv3'],'M']
    #     inputClassifier = networksDict[netConfig]['cnv3']
    # elif networksDict[netConfig]['cnv5']==0:
    #     featursParam = [networksDict[netConfig]['cnv1'],'M', networksDict[netConfig]['cnv2'],'M',networksDict[netConfig]['cnv3'],'M',networksDict[netConfig]['cnv4']]
    #     inputClassifier = networksDict[netConfig]['cnv4']
    # else:
    #     featursParam = [networksDict[netConfig]['cnv1'],     networksDict[netConfig]['cnv2'],'M',networksDict[netConfig]['cnv3'],'M',networksDict[netConfig]['cnv4'],'M',networksDict[netConfig]['cnv5'],32 ,32]
    #     inputClassifier = networksDict[netConfig]['cnv5']
    # creating convolution layers based on wanted size
    Net.features = Net.makeLayers(featursParam, inChannels)
    # # finding linear size
    # Net.fcSize = Net.get_flat_fts(inputSize, Net.features)
    # # creating linear layer
    # Net.classifier = nn.Sequential(nn.Conv2d(inputClassifier, classNum,kernel_size = 4,stride= 1,padding = 0,bias = False),nn.LogSoftmax())

    # setup network
    if isServerRun:
        Net = Net.cuda()
    numWeights = sum(param.numel() for param in Net.parameters())
    print('number of parameters: ', numWeights)

    # optimization parameters
    Net.lr = networksDict[netConfig]['lr']
    Net.optimizer = CreateOptimizer(Net.parameters(), networksDict[netConfig])
    Net.lossCrit = nn.NLLLoss()
    # network training parametrs
    Net.maxEpochs = maxEpochs
    Net.batchSize = networksDict[netConfig]['bs']

    # train
    print('training started...')
    startTime = time.clock()
    for numEpoch in range(Net.maxEpochs):
        localLoss = []
        accTrain = []
        trainCorr = 0.0
        trainTot = 0.0
        if (1+numEpoch)%20 == 0:
            if Net.optimizer.param_groups[0]['lr']>0.00001:
                Net.optimizer.param_groups[0]['lr'] = Net.optimizer.param_groups[0]['lr']/5
            else:
                Net.optimizer.param_groups[0]['lr'] = 0.00001
        print ('lr is: %.6f'%Net.optimizer.param_groups[0]['lr'])
        for i, (images, labels) in enumerate(dataloader_uber_train):
            # create torch variables
            imgVar = to_var(images)
            labVar = to_var(labels)
            # reset gradient
            Net.optimizer.zero_grad()
            # forward
            netOut = Net.forward(imgVar)
            # backwards
            Net.backward(netOut, labVar)
            # optimizer step
            Net.optimizer.step()
            # local loss function list
            localLoss.append(Net.loss.item())
            _, labTrain = torch.max(netOut.data, 1)
            if isServerRun:
                labTrain = labTrain.cpu()
            trainCorr = torch.sum(labTrain == labels).detach().numpy() + trainCorr
            trainTot = labels.size(0)*labels.size(1)*labels.size(2) + trainTot
            accTrain.append(100 * trainCorr / trainTot)
            # output current state
            if (i + 1) % 100 == 0:
                print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                       % (numEpoch + 1, Net.maxEpochs, i + 1,
                          dataloader_uber_train.dataset.data.shape[2] // dataloader_uber_train.batch_size,
                          Net.loss.item()))
        Net.lossVecTrain.append(np.average(localLoss))
        Net.accVecTrain.append(np.average(accTrain))
        # test network for each epoch stage
        Net.eval()
        accEpochTest, lossEpochTest = Net.test_spesific(testLoader=dataloader_uber_test)
        Net.accVecTest.append(accEpochTest)
        Net.lossVecTest.append(lossEpochTest)
    Net.finalAcc = accEpochTest
    Net.finalLoss = np.average(localLoss)
    endTime = time.clock()

    pickle.dump(Net, open('NetObj_' + netConfig + '.pkl', 'wn'))
    Net.saveModel(netConfig + ".pkl")
    # name, HyperPerams, accur, trainTime, num total weights
    # err vs epoch, loss vs epoch,
    strWrite = '{0};{1};{2};{3};{4};{5}\n'.format(netConfig, Net.finalAcc, Net.finalLoss, endTime - startTime,
                                                  numWeights, Net.maxEpochs)
    outFile.write(strWrite)

outFile.close()




