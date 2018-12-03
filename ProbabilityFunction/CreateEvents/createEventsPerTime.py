import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()


def createEventDistributionUber(startTime, endTime, probabilityMatrix, eventTimeWindow):
    eventPos = []
    eventTimes = []
    hour = [startTime // 4]  # each time is a quarter of an hour
    numTimeSteps = endTime - startTime
    if numTimeSteps > 4:
        hour.append(hour[0] + 1)
    for h in hour:
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = probabilityMatrix[x, y, h, :]
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')  # find how many events are happenning at the same time
                print('at loc:'+str(x)+','+str(y)+' num events:'+str(numEvents))
                for n in range(numEvents):
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(np.random.randint(0, 4) + h * 4)
    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    plt.scatter(eventsPos[:, 0], eventsPos[:, 1], label='event position')
    plt.legend()
    plt.figure()
    plt.scatter(range(eventTimes.shape[0]), eventTimes, label='times of event at hour:'+str(h))
    plt.legend()
    print('hi')
    return eventsPos, eventsTimeWindow


def main():
    starTime = 24
    endTime = 28
    probabilityMatrix = np.load('4DProbabilityCDF_wday_1.p')
    eventTimeWindow = 4
    eventPos,eventTime = createEventDistributionUber(starTime, endTime, probabilityMatrix, eventTimeWindow)

if __name__ == '__main__':
    main()
    print('done!')

