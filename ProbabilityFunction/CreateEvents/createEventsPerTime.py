import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


def createEventDistributionUber(startTime, endTime, probabilityMatrix, eventTimeWindow):
    eventPos = []
    eventTimes = []
    firstTime = startTime  # each time is a quarter of an hour
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = probabilityMatrix[x, y, t + firstTime, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                for n in range(numEvents):
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + firstTime)

    # for t in np.unique(np.array(eventTimes)):
    #     tempEventPos = [eventPos[i] for i in range(len(eventTimes)) if eventTimes[i] == t]
    #     for x, y in tempEventPos:
    #         plt.scatter(x, y)
    #         plt.text(x, y, 't=' + str(t))
    # plt.show()
    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    for i in range(endTime - startTime):
        numEventsPerTime = np.size(np.where(eventTimes==i))
        print('num events per time :'+str(i)+' is :'+str(numEventsPerTime))
    print('hi')
    return eventsPos, eventsTimeWindow


def main():
    starTime = 0
    endTime = 288
    probabilityMatrix = np.load('4D_UpdatedGrid_5min_LimitedProbability_CDFMat_wday_1.p')
    eventTimeWindow = 4
    eventPos,eventTime = createEventDistributionUber(starTime, endTime, probabilityMatrix, eventTimeWindow)
    print('done..')


if __name__ == '__main__':
    main()
    print('done!')

