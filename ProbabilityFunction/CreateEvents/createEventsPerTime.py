import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()







def createEventDistributionUber(startTime, endTime, probabilityMatrix, eventTimeWindow):
    eventPos = []
    eventTimes = []
    hour = [startTime % 4]  # each time is a quarter of an hour
    numTimeSteps = endTime - startTime
    if numTimeSteps > 4:
        hour.append(hour[0] + 1)
    for h in hour:
        for x, y in zip(range(probabilityMatrix.shape[0]), range(probabilityMatrix.shape[1])):
            randNum = np.random.uniform(0, 1)
            if randNum <= probabilityMatrix[x, y, h]:
                eventPos.append([x, y])
                eventTimes.append(4 * h + np.random.randint(0, 4))

    eventsPos = np.array(eventPos)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsPos, eventsTimeWindow


def main():
    TimeMatrix = np.load('3DMat_weekNum_18wday_2.p')
    starTime = 0
    endTime = 4
    probabilityMatrix = np.load('weekNum18_wday2_probabilityMatrix.p')
    eventTimeWindow = 4
    eventPos,eventTime = createEventDistributionUber(starTime,endTime,probabilityMatrix,eventTimeWindow)

    print('hi')


if __name__ == '__main__':
    main()
    print('done!')

