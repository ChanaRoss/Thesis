import numpy as np
import torch
from matplotlib import pyplot as plt



class DataSet_maxFlow:
    def __init__(self, dataIn, lenSeqIn, numCars, shouldFlatten):
        self.lengthX    = dataIn.shape[0]
        self.lengthY    = dataIn.shape[1]
        self.lengthT    = dataIn.shape[2]
        self.numCars    = numCars
        data            = np.swapaxes(dataIn, 0, 1)  # swap x and y axis , now matrix size is: [y, x, t]
        data            = np.swapaxes(data, 0, 2)    # swap y and t axis , now matrix size is: [t, x, y]
        self.data       = data
        self.lenSeqIn   = lenSeqIn
        self.numCars    = numCars
        self.shouldFlatten = shouldFlatten

    def __getitem__(self, item):
        temp = np.zeros(shape=(self.lenSeqIn, self.lengthX, self.lengthY))
        if (item - self.lenSeqIn > 0):
            temp = self.dataLabel[item - self.lenSeqIn:item, :, :]
        else:
            temp[self.lenSeqIn - item:, :, :] = self.dataLabel[0:item, :, :]

        if self.shouldFlatten:
            xArr = np.zeros(shape=(self.lengthX*self.lengthY, self.lenSeqIn))
            k = 0
            for i in range(self.lengthX):
                for j in range(self.lengthY):
                    xArr[k, :]  = temp[:, i, j]
                    k += 1
        else:
            xArr = temp

        if torch.cuda.is_available():
            xTensor = torch.Tensor(xArr).cuda()
        else:
            xTensor = torch.Tensor(xArr)
        # xTensor is of shape: [grid id, seq] or [x, y, seq] (Depends if flatten is true or false
        # xTensor_cars: [2, numCars]
        # yTensor: 1 (this is the value of the max flow run)
        return xTensor

    def __len__(self):
        return self.data.shape[0]


def main():
    file_loc = '/Users/chanaross/dev/Thesis/UberData/'
    fileNameDist = '4D_ProbabilityMat_allDataLatLonCorrected_20MultiClass_CDF_500gridpickle_30min.p'
    return


if __name__ == '__main__':
    main()
    print('Done.')
