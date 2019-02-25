import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import sys,os
sns.set()






TimeProbMatrix = np.load('/Users/chanaross/dev/Thesis/UberData/4D_UpdatedGrid_5min_250Grid_LimitedProbabilityMat_allData_Benchmark.p')
cdf = np.zeros(TimeProbMatrix.shape[3])
FullCdf = np.zeros(shape = (TimeProbMatrix.shape[0], TimeProbMatrix.shape[1], TimeProbMatrix.shape[2], cdf.size))

# xSteps = np.arange(10, 30, 1)
# ySteps = np.arange(20, 40, 1)
# timeSteps = np.arange(100, 110, 1)
for t in range(TimeProbMatrix.shape[2]):
    for ix in range(TimeProbMatrix.shape[0]):
        for iy in range(TimeProbMatrix.shape[1]):
            probMatrix = TimeProbMatrix[ix, iy, t, :].reshape((cdf.size, 1))
            for i in range(probMatrix.size):
                cdf[i] = np.sum(probMatrix[0:i+1])
            # plt.figure()
            # plt.plot(range(probMatrix.size), cdf, label='cdf of Time:' + str(t)+' loc:'+str(ix)+','+str(iy))
            # plt.ylabel('CDF - number of events per time and position')
            # plt.xlabel('number of events')
            # plt.legend()
            # plt.show()
            FullCdf[ix, iy, t, :] = cdf
    # plt.figure()
    # plt.matshow(np.sum(TimeProbMatrix[:,:,t,:],axis = 2))
    # plt.show()
FullCdf.dump('4D_UpdatedGrid_5min_250Grid_LimitedProbability_CDF_mat_allData_Benchmark.p')

print("done!")