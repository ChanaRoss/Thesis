# mathematical imports -
import numpy as np

# pytorch imports  -
import torch
import torch.utils.data as data

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def createDiff(data):
    dataOut = np.zeros(shape=(data.shape[0],data.shape[1]-1))
    for i in range(data.shape[1]-1):
        dataOut[:, i] = data[:, i+1] - data[:, i]
    return dataOut



class DataSetUber:
    def __init__(self, dataIn, lenSeqIn, lenSeqOut):
        self.data = dataIn
        self.lenSeqIn = lenSeqIn
        self.lenSeqOut = lenSeqOut

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.data.shape[0], self.lenSeqIn))
        if(item-self.lenSeqIn > 0):
            temp = self.data[:, item-self.lenSeqIn:item]
        else:
            temp[:, self.lenSeqIn-item:] = self.data[:, 0:item]

        tempOut = np.zeros(shape=(self.data.shape[0], self.lenSeqOut))
        try:
            if item + 1 + self.lenSeqOut < self.data.shape[1]:
                tempOut = self.data[:, item+1:item+1+self.lenSeqOut].reshape(self.data.shape[0], self.lenSeqOut)
            else:
                numFuture = self.data.shape[1] - (item+1)
                tempOut[:, 0:numFuture] = self.data[:, item+1:]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')
        return torch.Tensor(temp, device=device), torch.Tensor(tempOut, device=device)


    def __len__(self):
        return self.data.shape[1]



class DataSetLstm:
    def __init__(self, dataIn, lenSeqIn):
        self.data = dataIn
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.data.shape[0], self.lenSeqIn))
        if(item-self.lenSeqIn > 0):
            temp = self.data[:, item-self.lenSeqIn:item]
        else:
            temp[:, self.lenSeqIn-item:] = self.data[:, 0:item]

        tempOut = np.zeros(shape=(self.data.shape[0], self.lenSeqIn))
        try:

            if   (item + 1 <= self.data.shape[1]) and (item + 1 - self.lenSeqIn > 0):
                tempOut = self.data[:, item + 1 - self.lenSeqIn: item + 1].reshape(self.data.shape[0], self.lenSeqIn)
            elif (item + 1 <= self.data.shape[1]) and (item + 1 - self.lenSeqIn < 0):
                tempOut[:, self.lenSeqIn - item - 1:] = self.data[:, 0:item + 1]
            elif (item + 1 > self.data.shape[1]) and (item + 1 - self.lenSeqIn > 0):
                tempOut[:, 0:self.lenSeqIn-1] = self.data[:, item + 1 - self.lenSeqIn: item]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')
        return torch.Tensor(temp, device=device), torch.Tensor(tempOut, device=device)

    def __len__(self):
        return self.data.shape[1]




class DataSetCnn:
    def __init__(self, dataIn, lenSeqIn):
        self.lengthX = dataIn.shape[0]
        self.lengthY = dataIn.shape[1]
        self.lengthT = dataIn.shape[2]
        self.data = dataIn.reshape(self.lengthT, self.lengthX, self.lengthY)
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        xArr = temp
        tempOut = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        try:

            if (item + 1 <= self.data.shape[0]) and (item + 1 - self.lenSeqIn > 0):
                tempOut = self.data[item + 1 - self.lenSeqIn: item + 1, :, :].reshape(self.lenSeqIn, self.data.shape[1], self.data.shape[2])
            elif (item + 1 <= self.data.shape[0]) and (item + 1 - self.lenSeqIn <= 0):
                tempOut[self.lenSeqIn - item - 1:, :, :] = self.data[0:item + 1, :, :]
            elif (item + 1 > self.data.shape[0]) and (item + 1 - self.lenSeqIn > 0):
                tempOut[0:self.lenSeqIn - 1, :, :] = self.data[item + 1 - self.lenSeqIn: item, :, :]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')
        try:
            yArr = tempOut[-1, :, :]
        except:
            print("couldnt take last value of time sequence for output!!!")
        return torch.Tensor(xArr, device=device), torch.Tensor(yArr, device=device).type(torch.long)

    def __len__(self):
        return self.data.shape[0]

class DataSetCnn_LSTM:
    def __init__(self, dataIn, lenSeqIn, sizeCnn):
        self.lengthX = dataIn.shape[0]
        self.lengthY = dataIn.shape[1]
        self.lengthT = dataIn.shape[2]
        self.sizeCnn = sizeCnn
        self.data = dataIn.reshape(self.lengthT, self.lengthX, self.lengthY)
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        temp2 = np.zeros(shape=(self.lenSeqIn, self.sizeCnn , self.sizeCnn, self.lengthX*self.lengthY))
        tempPadded = np.zeros(shape=(temp.shape[0], temp.shape[1]+self.sizeCnn, temp.shape[2]+self.sizeCnn))
        tempPadded[:, self.sizeCnn: self.sizeCnn + temp.shape[1],  self.sizeCnn : self.sizeCnn + temp.shape[2]] = temp
        k = 0
        for i in range(self.lengthX):
            for j in range(self.lengthY):
                try:
                    temp2[:, :, :, k] = tempPadded[:, i:i + self.sizeCnn, j : j+self.sizeCnn]
                except:
                    print("couldnt create input for cnn ")
                k += 1
        xArr = temp2
        tempOut = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        try:

            if (item + 1 <= self.data.shape[0]) and (item + 1 - self.lenSeqIn > 0):
                tempOut = self.data[item + 1 - self.lenSeqIn: item + 1, :, :].reshape(self.lenSeqIn, self.data.shape[1], self.data.shape[2])
            elif (item + 1 <= self.data.shape[0]) and (item + 1 - self.lenSeqIn <= 0):
                tempOut[self.lenSeqIn - item - 1:, :, :] = self.data[0:item + 1, :, :]
            elif (item + 1 > self.data.shape[0]) and (item + 1 - self.lenSeqIn > 0):
                tempOut[0:self.lenSeqIn - 1, :, :] = self.data[item + 1 - self.lenSeqIn: item, :, :]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')

        try:
            yArr = tempOut[-1, :, :]
        except:
            print("couldnt take last value of time sequence for output!!!")


        if torch.cuda.is_available():
            xTensor = torch.Tensor(xArr).cuda()
            yTensor = torch.Tensor(yArr).type(torch.cuda.LongTensor)
        else:
            xTensor = torch.Tensor(xArr)
            yTensor = torch.Tensor(yArr).type(torch.long)
        return xTensor, yTensor

    def __len__(self):
        return self.data.shape[0]


def main():
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName  = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_allData.p'
    dataInput = np.load(path + fileName)
    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20
    dataInput = dataInput[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model
    # define important sizes for network -
    x_size = dataInput.shape[0]
    y_size = dataInput.shape[1]
    dataSize = dataInput.shape[2]
    num_train = int((1 - 0.2) * dataSize)
    data_train = dataInput[:, :, 0:num_train]
    dataset_uber = DataSetCnn_LSTM(data_train, 5,  7)
    dataloader_uber = data.DataLoader(dataset=dataset_uber, batch_size=300, shuffle=True)
    # a = list(iter(dataset_uber))

    for i_batch, sample_batched in enumerate(dataloader_uber):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size())
    return







if __name__ == '__main__':
    main()
    print('Done.')
