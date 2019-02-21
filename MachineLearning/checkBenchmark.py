import numpy as np
from matplotlib import  pyplot as plt



def createEventDistributionUber(simStartTime, startTime, endTime, probabilityMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = probabilityMatrix[x, y, t + firstTime, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                #for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos  = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:'+str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow

def createRealEventsDistributionUber(simStartTime, startTime, endTime, eventsMatrix,eventTimeWindow,simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        for x in range(eventsMatrix.shape[0]):
            for y in range(eventsMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                numEvents = eventsMatrix[x, y, t + firstTime]
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                # for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    print('number of events created:' + str(eventsPos.shape[0]))
    return eventsPos, eventsTimeWindow


def main():
    # data loader -
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_allData.p'
    dataInput = np.load(path + fileName)

    xmin = 0
    xmax = 20
    ymin = 0
    ymax = 20
    zmin = 4000
    zmax = 4601
    dataInput = dataInput[xmin:xmax, ymin:ymax,zmin:zmax]  # shrink matrix size for fast training in order to test model





    return




if __name__ == '__main__':
    main()
    print('Done.')