import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import optimize




class Predictor():
    def __init__(self,mat):
        self.mat = mat
        self.lmda = None
        self.probFunc = None

    def getFeatures(self,indexX,indexY,indexT):
        featuresOut = []
        tempResult = self.calculateFeature1(indexX,indexY,indexT)
        featuresOut.append(tempResult)
        tempResult2 = self.calculateFeature2(indexX, indexY, indexT)
        featuresOut.append(tempResult2)
        tempResult3 = self.calculateFeature3(indexX, indexY, indexT)
        featuresOut.append(tempResult3)
        tempResult4 = self.calculateFeature4(indexX, indexY, indexT)
        featuresOut.append(tempResult4)
        return featuresOut

    def calculateFeature1(self,indexX,indexY,indexT):
        # this feature returns the amount of events in the grid location at time T
        return self.mat[indexX,indexY,indexT]

    def calculateFeature2(self, indexX,indexY,indexT):
        # this function returns the sum of all events in the same grid row at time T
        result = np.sum(self.mat[indexX, :,indexT])
        return result

    def calculateFeature3(self, indexX,indexY,indexT):
        # this function returns the sum of all events in the same grid column at time T
        result = np.sum(self.mat[:, indexY, indexT])
        return result

    def calculateFeature4(self, indexX,indexY,indexT):
        # this function returns the sum of all events in the same grid location at all times before T
        matPadded = np.pad(self.mat, 1, mode='constant')
        result = np.sum(matPadded[indexX+1, indexY+1, 0:indexT+1])
        return result



    def optimizationFunction(self,lmda):
        z_lmda = 0
        # mat = args[0] # written this way for optimize function in scipy
        matSum = np.sum(self.mat.astype(np.int64))
        assert (matSum != 0)
        tempSum = 0
        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                for k in range(self.mat.shape[2]):
                    feature_out = self.getFeatures(i, j, k)
                    dotProductFeatureLmda = np.dot(lmda, feature_out)
                    tempSum += self.mat[i][j][k] * dotProductFeatureLmda
                    z_lmda += np.exp(dotProductFeatureLmda)
        tempSum = float(tempSum) / matSum
        result = np.log(z_lmda) - tempSum
        return result

    def pFound(self,lmda, indexX, indexY,indexT, z_lmda):
        result = np.exp(np.dot(lmda, self.getFeatures(indexX, indexY,indexT))) / z_lmda
        return result

    def fit(self):
        numFeatures = 4
        lmda0 = np.zeros(shape=(1, numFeatures)) + 0.004
        matX = self.mat.shape[0]
        matY = self.mat.shape[1]
        matT = self.mat.shape[2]
        result = optimize.minimize(self.optimizationFunction, x0=lmda0, method='Powell', tol=0.01, options={'maxiter': 400})
        lmdaOpt = result.x
        print(result.message)
        print(lmdaOpt)
        matSum = np.sum(self.mat)
        pNumeric = self.mat / matSum
        pFound = np.zeros_like(pNumeric)
        z_lmda = 0
        for i in range(matX):
            for j in range(matY):
                for k in range(matT):
                    z_lmda += np.exp(np.dot(lmdaOpt, self.getFeatures(i, j,k)))
        for i in range(matX):
            for j in range(matY):
                for k in range(matT):
                    pFound[i][j][k] = self.pFound(lmdaOpt, i, j,k, z_lmda)
        norm1 = np.linalg.norm(pFound.reshape(1, pFound.size) - pNumeric.reshape(1, pNumeric.size), 1)
        print('norm1 is:' + str(norm1))
        for i in range(matT):
            fig, axs = plt.subplots(1, 2, facecolor='w', edgecolor='k')
            axs[0].matshow(pNumeric[:,:,i])
            axs[1].matshow(pFound[:,:,i])
        plt.show()
        return pFound,lmdaOpt

    def update(self,mat):
        self.mat = mat

    def updateProbabilityFunction(self,mat):
        self.update(mat)
        self.probFunc,self.lmda = self.fit()


    def sample(self,n):
        pass


def main():
    mat3D  = np.load('/home/chana/Documents/Thesis/Uber_Analysis/PostAnalysis/3DMat_weekNum_18wday_2.p')
    # mat = np.random.randint(0,22,size=[4,5,3])
    predictor = Predictor(mat = mat3D)
    prob,lmda = predictor.fit()
    print('done')



if __name__ == '__main__':
    main()
    print('im done')