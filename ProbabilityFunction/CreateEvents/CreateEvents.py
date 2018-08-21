import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

sns.set()



TimeMatrix = np.load('3DMat_weekNum_18wday_2.p')



TimeProbMatrix = np.load('weekNum18_wday2_probabilityMatrix.p')
cdf = np.zeros(TimeProbMatrix[:,:,1].size)
TimeCdf = np.zeros(shape = (cdf.size,24))
accumalativeCdf = 0
numEventsPerHour = np.zeros(24)
for time in [13]:
    numEventsPerHour[time] = np.sum(TimeMatrix[:,:,time])
    probMatrix = TimeProbMatrix[:,:,time].reshape((TimeProbMatrix[:,:,time].size,1))
    for i in range(probMatrix.size):
        if i == 857:
            print('stop')
        cdf[i] = np.sum(probMatrix[0:i+1])
    TimeCdf[:,time] = cdf
    print('Num Events for Time : '+str(time) + ' is: ' + str(numEventsPerHour[time]))

    plt.matshow(TimeProbMatrix[:,:,time])
    plt.figure()
    plt.matshow(TimeMatrix[:,:,time]/np.sum(TimeMatrix[:,:,time]))
    plt.figure()
    plt.plot(range(probMatrix.size),probMatrix)
    plt.figure()
    plt.plot(range(probMatrix.size),cdf,label= 'cdf of Time:' + str(i))
    plt.xlabel('grid id')
    plt.ylabel('CFD')
plt.show()

