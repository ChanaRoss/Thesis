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
        return self.data.shape[2]




class DataSetCnn:
    def __init__(self, dataIn, lenSeqIn):
        self.data = dataIn
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.data.shape[0], self.data.shape[1], self.lenSeqIn))
        if (item - self.lenSeqIn > 0):
            temp = self.data[:, :, item - self.lenSeqIn:item]
        else:
            temp[:, :, self.lenSeqIn - item:] = self.data[:, :, 0:item]
        xArr = temp
        tempOut = np.zeros(shape=(self.data.shape[0], self.data.shape[1], self.lenSeqIn))
        try:

            if (item + 1 <= self.data.shape[2]) and (item + 1 - self.lenSeqIn > 0):
                tempOut = self.data[:, :, item + 1 - self.lenSeqIn: item + 1].reshape(self.data.shape[0],self.data.shape[1], self.lenSeqIn)
            elif (item + 1 <= self.data.shape[2]) and (item + 1 - self.lenSeqIn <= 0):
                tempOut[:, :, self.lenSeqIn - item - 1:] = self.data[:, :, 0:item + 1]
            elif (item + 1 > self.data.shape[2]) and (item + 1 - self.lenSeqIn > 0):
                tempOut[:, :, 0:self.lenSeqIn - 1] = self.data[:, :, item + 1 - self.lenSeqIn: item]  # taking the last part of the sequence
        except:
            print('couldnt find correct output sequence!!!')
        try:
            yArr = tempOut[:, :, -1]
        except:
            print("couldnt take last value of time sequence for output!!!")
        return torch.Tensor(xArr, device=device), torch.Tensor(yArr, device=device).type(torch.long)

    def __len__(self):
        return self.data.shape[2]





def main():
    path      = '/home/chanaby/Documents/Thesis/machineLearning/'
    fileName  = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_wday_1.p'
    dataInput = np.load(path + fileName)
    dataTemp  = dataInput.reshape(dataInput.shape[0]*dataInput.shape[1], dataInput.shape[2])
    dataDiff  = createDiff(dataTemp)


    dataset_uber = DataSetLstm(dataDiff, 10)
    dataloader_uber = data.DataLoader(dataset=dataset_uber ,batch_size=30, shuffle=True)
    # a = list(iter(dataset_uber))

    for i_batch, sample_batched in enumerate(dataloader_uber):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size())
    return







if __name__ == '__main__':
    main()
    print('Done.')
