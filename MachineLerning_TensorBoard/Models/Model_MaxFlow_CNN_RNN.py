import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn            = nn.ModuleList()
        self.fc_after_cnn   = nn.ModuleList()
        self.lstm           = None
        self.fc_after_lstm  = None
        self.logSoftMax     = nn.LogSoftmax()


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
        cnn_output = torch.zeros([self.batch_size, self.fc_output_size, self.sequence_size])
        # x is of size : [batch_size , mat_x , mat_y , sequence_size]
        for i in range(self.sequence_size):
            batch_size = x.size(0)
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
        self.loss = self.lossCrit(outputs, labels)
        # if self.loss is None:
        #     self.loss = self.lossCrit(outputs, labels)
        # else:
        #     self.loss += self.lossCrit(outputs, labels)

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
        for inputs, labels in testLoader:
            inputVar = to_var(inputs)
            labVar = to_var(labels)
            # compute test result of model
            x_size = inputVar.shape[2]
            y_size = inputVar.shape[3]
            testOut = np.zeros(shape=(self.batch_size, self.class_size, x_size, y_size))
            k = 0
            for x in range(x_size):
                for y in range(y_size):  # calculate output for each grid_id
                    testOut[:, :, x, y] = self.forward(inputVar[:, :, :, :, k])
                    k += 1
            # find loss of test set
            self.backward(testOut, labVar)
            localLossTest.append(self.loss.item())
            _, labTest = torch.max(testOut.data, 1)
            if torch.cuda.is_available():
                labTest = labTest.cpu()
            testCorr = torch.sum(labTest == labels).detach().numpy() + testCorr
            testTot = labels.size(0) * labels.size(1) * labels.size(2) + testTot
            localAccTest.append(100 * testCorr / testTot)
        accTest = np.average(localAccTest)
        lossTest = np.average(localLossTest)
        print("test accuarcy is: {0}".format(accTest))
        return accTest, lossTest

        # save network

    def saveModel(self, path):
        torch.save(self, path)

