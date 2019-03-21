# mathematical imports -
import numpy as np
from matplotlib import pyplot as plt
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
        data = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data = np.swapaxes(data, 0, 2)  # swap y and t axis , now matrix size is: [t, x, y]
        self.data = data
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
        data = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data = np.swapaxes(data, 0, 2)  # swap y and t axis , now matrix size is: [t, x, y]
        self.data = data
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        temp2 = np.zeros(shape=(self.lenSeqIn, self.sizeCnn, self.sizeCnn, self.lengthX*self.lengthY))
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


class DataSetCnn_LSTM_BatchMode:
    def __init__(self, dataIn, lenSeqIn, sizeCnn):
        self.lengthX = dataIn.shape[0]
        self.lengthY = dataIn.shape[1]
        self.lengthT = dataIn.shape[2]
        self.sizeCnn = sizeCnn
        data = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data = np.swapaxes(data, 0, 2)  # swap y and t axis , now matrix size is: [t, x, y]
        self.data = data
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        temp2 = np.zeros(shape=(self.lengthX*self.lengthY, self.lenSeqIn, self.sizeCnn, self.sizeCnn))
        tempPadded = np.zeros(shape=(temp.shape[0], temp.shape[1]+self.sizeCnn, temp.shape[2]+self.sizeCnn))
        tempPadded[:, self.sizeCnn: self.sizeCnn + temp.shape[1],  self.sizeCnn : self.sizeCnn + temp.shape[2]] = temp
        k = 0
        for i in range(self.lengthX):
            for j in range(self.lengthY):
                try:
                    temp2[k, :, :, :] = tempPadded[:, i:i + self.sizeCnn, j: j+self.sizeCnn]
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
        # xTensor is of shape: [grid id, seq, x_cnn, y_cnn]
        # yTensor is of shape: [grid x, grid y]
        return xTensor, yTensor

    def __len__(self):
        return self.data.shape[0]


class DataSetCnn_LSTM_NonZero:
    def __init__(self, dataIn, lenSeqIn, sizeCnn, max_to_add):
        self.lengthX = dataIn.shape[0]
        self.lengthY = dataIn.shape[1]
        self.lengthT = dataIn.shape[2]
        self.sizeCnn = sizeCnn
        self.max_to_add = max_to_add
        data = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data = np.swapaxes(data, 0, 2)  # swap y and t axis , now matrix size is: [t, x, y]
        self.data = data
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        tempPadded = np.zeros(shape=(temp.shape[0], temp.shape[1]+self.sizeCnn, temp.shape[2]+self.sizeCnn))
        padding_loc = np.floor_divide(self.sizeCnn,2)
        tempPadded[:, padding_loc: padding_loc + temp.shape[1], padding_loc: padding_loc + temp.shape[2]] = temp
        # full_input_matrix is of size: [seq_len, size_cnn, size_cnn, grid_x*grid_y]
        # creating input to cnn network for each grid point -
        full_input_matrix   = np.zeros(shape=(self.lenSeqIn, self.sizeCnn, self.sizeCnn, self.lengthX*self.lengthY))
        actual_input_matrix = np.zeros(shape=(self.lenSeqIn, self.sizeCnn, self.sizeCnn, self.max_to_add))
        k = 0
        x_indices = np.zeros(shape=(self.lengthY*self.lengthX))
        y_indices = np.zeros(shape=(self.lengthY*self.lengthX))
        for i in range(self.lengthX):
            for j in range(self.lengthY):
                mat_to_add = tempPadded[:, i:i + self.sizeCnn, j: j+self.sizeCnn]
                full_input_matrix[:, :, :, k] = mat_to_add
                x_indices[k] = i
                y_indices[k] = j
                k += 1

        # choosing most relevant cnn inputs-
        non_zero_indices = np.where(np.sum(full_input_matrix[-1, :, :, :], axis=(0, 1)) != 0)[0]
        zero_indices     = np.where(np.sum(full_input_matrix[-1, :, :, :], axis=(0, 1)) == 0)[0]
        if non_zero_indices.shape[0] >= self.max_to_add:
            np.random.shuffle(non_zero_indices)
            actual_input_matrix = full_input_matrix[:, :, :, non_zero_indices[0:self.max_to_add]]
            x_index_output = x_indices[non_zero_indices[0:self.max_to_add]].astype(int)
            y_index_output = y_indices[non_zero_indices[0:self.max_to_add]].astype(int)
        else:
            np.random.shuffle(non_zero_indices)
            np.random.shuffle(zero_indices)
            actual_input_matrix[:, :, :, 0:non_zero_indices.shape[0]]   = full_input_matrix[:, :, :, non_zero_indices]
            num_zeros_to_add = self.max_to_add - non_zero_indices.shape[0]
            actual_input_matrix[:, :, :, non_zero_indices.shape[0]:] = full_input_matrix[:, :, :, zero_indices[0:num_zeros_to_add]]
            x_index_output = x_indices[np.concatenate([non_zero_indices[0:self.max_to_add], zero_indices[0:num_zeros_to_add]])].astype(int)
            y_index_output = y_indices[np.concatenate([non_zero_indices[0:self.max_to_add], zero_indices[0:num_zeros_to_add]])].astype(int)

        xArr = actual_input_matrix
        tempOut = np.zeros(self.max_to_add)
        for i in range(self.max_to_add):
            if item < self.lengthT:
                tempOut[i] = self.data[item, x_index_output[i], y_index_output[i]]

        yArr = tempOut

        if torch.cuda.is_available():
            xTensor = torch.Tensor(xArr).cuda()
            yTensor = torch.Tensor(yArr).type(torch.cuda.LongTensor)
        else:
            xTensor = torch.Tensor(xArr)
            yTensor = torch.Tensor(yArr).type(torch.long)
        return xTensor, yTensor

    def __len__(self):
        return self.data.shape[0]


class DataSet_oneLSTM_allGrid:
    def __init__(self, dataIn, lenSeqIn):
        self.lengthX = dataIn.shape[0]
        self.lengthY = dataIn.shape[1]
        self.lengthT = dataIn.shape[2]
        data = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data = np.swapaxes(data, 0, 2)    # swap y and t axis , now matrix size is: [t, x, y]
        self.data = data
        self.lenSeqIn = lenSeqIn

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.data.shape[1], self.data.shape[2]))
        if (item - self.lenSeqIn > 0):
            temp = self.data[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.data[0:item, :, :]
        xArr = np.zeros(shape=(self.lengthX*self.lengthY, self.lenSeqIn))
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
            yArrTemp = tempOut[-1, :, :]
        except:
            print("couldnt take last value of time sequence for output!!!")

        yArr = np.zeros(shape=(self.lengthY*self.lengthX))
        k = 0
        for i in range(self.lengthX):
            for j in range(self.lengthY):
                xArr[k, :]  = temp[:, i, j]
                yArr[k]     = yArrTemp[i, j]
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

    def __len__(self):
        return self.data.shape[0]


def main():
    # path = '/home/chanaby/Documents/dev/Thesis/MachineLearning/forGPU/'
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName  = '3D_allDataLatLonCorrected_binaryClass_500gridpickle_30min.p'
    dataInput = np.load(path + fileName)
    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20
    zmin = 48
    dataInput = dataInput[xmin:xmax, ymin:ymax, zmin:]  #  shrink matrix size for fast training in order to test model
    # dataInput = dataInput[8:10, 12:14, 16000:32000].reshape((2,2,32000-16000))  # shrink matrix size for fast training in order to test model

    # define important sizes for network -
    x_size = dataInput.shape[0]
    y_size = dataInput.shape[1]
    dataSize = dataInput.shape[2]
    num_train = int((1 - 0.2) * dataSize)
    data_train = dataInput[:, :, 0:num_train]
    dataset_uber = DataSet_oneLSTM_allGrid(data_train, 10)
    dataloader_uber = data.DataLoader(dataset=dataset_uber, batch_size=300, shuffle=True)
    # a = list(iter(dataset_uber))

    for i_batch, sample_batched in enumerate(dataloader_uber):
        print(i_batch, sample_batched[0].size(),
              sample_batched[1].size())
    return







if __name__ == '__main__':
    main()
    print('Done.')
