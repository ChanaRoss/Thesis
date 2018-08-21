import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import optimize




class Predictor():
    def __init__(self,mat):
        self.mat = mat
        self.lmda = None
        self.probFunc = None
        self.z_lmda_temp = 0

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
        self.z_lmda_temp = z_lmda
        return result

    def optimizationGradient(self,lmda):

        # mat = args[0] # written this way for optimize function in scipy
        matSum = np.sum(self.mat.astype(np.int64))
        assert (matSum != 0)
        tempSum = np.zeros(shape=(1,lmda.size))
        z_lmdaVector = np.zeros(shape=(1,lmda.size))
        for i in range(self.mat.shape[0]):
            for j in range(self.mat.shape[1]):
                for k in range(self.mat.shape[2]):
                    feature_out = self.getFeatures(i, j, k)
                    productFeatureLmda = np.dot(lmda, feature_out)
                    tempSum =np.add(tempSum,np.dot(self.mat[i,j,k],feature_out))
                    z_lmdaVector  = np.add(z_lmdaVector, np.multiply(np.exp(productFeatureLmda),feature_out))
        tempSum = tempSum/matSum
        result = z_lmdaVector/self.z_lmda_temp - tempSum
        return result

    def pFound(self,lmda, indexX, indexY,indexT, z_lmda):
        result = np.exp(np.dot(lmda, self.getFeatures(indexX, indexY,indexT))) / z_lmda
        return result

    def fit(self):
        numFeatures = 4
        lmda0 = np.zeros(shape=(1, numFeatures)) #+ 0.004
        matX = self.mat.shape[0]
        matY = self.mat.shape[1]
        matT = self.mat.shape[2]
        flag_StopCalc = False
        lmda = [lmda0]
        f = [self.optimizationFunction(lmda0)]
        lr = 0.001
        epsilon = 0.0001
        for n in range(300):
            if flag_StopCalc==True:
                print("finished optimization")
                break
            else:
                lmda_dot = self.optimizationGradient(lmda[-1])
                # lmda_dot = self.checkGradient(lmda[-1])
                lmda_temp = lmda[-1] - lr*lmda_dot
                lmda.append(lmda_temp)
                print("iteration number:" + str(n))
                f.append(self.optimizationFunction(lmda[-1]))
                deltaF = abs(f[-1]-f[-2])
                print(deltaF,lmda[-1])
                if deltaF<epsilon:
                    flag_StopCalc = True
        lmdaArray = np.vstack(lmda)
        fig,axs = plt.subplots(2,2,facecolor = 'w')
        axs[0][0].plot(range(lmdaArray.shape[0]),lmdaArray[:,0])
        axs[0][1].plot(range(lmdaArray.shape[0]), lmdaArray[:, 1])
        axs[1][0].plot(range(lmdaArray.shape[0]), lmdaArray[:, 2])
        axs[1][1].plot(range(lmdaArray.shape[0]), lmdaArray[:, 3])
        lmdaOpt = lmda[-1]
        print(f[-1])

        # result = optimize.minimize(self.optimizationFunction, x0=lmda0, method='Powell', tol=0.01, options={'maxiter': 400})
        # lmdaOpt = result.x
        # print(result.message)
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

    def checkGradient(self,lmda):
        f1 = self.optimizationFunction(lmda)
        grad = np.zeros(4)
        for i in range(4):
            lmda2 = np.array(lmda)
            lmda2[0,i] = lmda[0,i]+0.0002
            f2 = self.optimizationFunction(lmda2)
            grad[i] = (f2 - f1)/0.0002
        return grad


def main():
    # mat3D  = np.load('/home/chana/Documents/Thesis/Uber_Analysis/PostAnalysis/3DMat_weekNum_18wday_2.p')
    mat3D = np.random.randint(0,22,size=[15,15,8])
    # mat3D = mat3D[20:60,0:44,:]
    predictor = Predictor(mat = mat3D)
    # lmda = np.zeros([100,4])
    # lmda[:,0] = np.linspace(0,1,100)
    # f = []
    # gr = []
    # grNum = []
    # for lm in lmda:
    #     print(lm)
    #     f.append(predictor.optimizationFunction(lm))
    #     gr.append(predictor.optimizationGradient(lm))
    #     grNum.append(predictor.checkGradient(lm))
    # fAr = np.vstack(f)
    # grAr = np.vstack(gr)
    # grNumAr = np.vstack(grNum)
    # plt.figure()
    # plt.plot(lmda[:,0],fAr)
    # plt.figure()
    # plt.plot(lmda[:,0],grAr[:,0])
    # plt.plot(lmda[:,0],grNumAr)
    prob,lmda = predictor.fit()
    print('done')



if __name__ == '__main__':
    main()
    print('im done')