import numpy as np
from scipy.stats import truncnorm
import torch


def poissonRandomEvents(startTime, endSimTime, lam):
    """
    creates time line assuming poisson distribution
    :param startTime: start time wanted for timeline
    :param endSimTime: end time wanted for timeline
    :param lam: lambda of poisson distribution (how often will an event appear?)
    :return: list of times for events
    """
    numTimeSteps = int(endSimTime - startTime)
    numEventsPerTime = np.random.poisson(lam=lam, size=numTimeSteps)
    eventTime = []
    for i in range(numTimeSteps):
        nTime = numEventsPerTime[i]
        if nTime>0:
            for num in range(nTime):
                eventTime.append(i+startTime)
    return np.array(eventTime)


def createEventsDistribution(gridSize, startTime, endTime, lam, eventTimeWindow):
    eventsTimes = create_events_times(startTime, endTime, lam, eventTimeWindow)
    eventsPos = create_events_position(gridSize, eventsTimes.shape[0])
    return eventsPos, eventsTimes


def create_events_times(start_time, end_time, lam, eventTimeWindow):
    # randomize event times
    eventTimes = poissonRandomEvents(start_time, end_time, lam)
    eventsTimes = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsTimes


def create_events_position(gridSize, n_points):
    locX = gridSize[0] / 2
    scaleX = gridSize[0] / 2
    locY = gridSize[1] / 2
    scaleY = gridSize[1] / 2

    # locX = gridSize[0] / 2
    # scaleX = gridSize[0] / 5
    # locY = gridSize[1] / 10
    # scaleY = gridSize[1] / 5
    eventPosX = truncnorm.rvs((0 - locX) / scaleX, (gridSize[0] - locX) / scaleX, loc=locX, scale=scaleX,
                              size=n_points).astype(np.int64)
    eventPosY = truncnorm.rvs((0 - locY) / scaleY, (gridSize[1] - locY) / scaleY, loc=locY, scale=scaleY,
                              size=n_points).astype(np.int64)
    eventsPos = np.column_stack([eventPosX, eventPosY])
    return eventsPos


def createEventDistributionMl(simstartTime, startTime, endTime, previousMat, eventTimeWindow, simTime, my_net):
    """
    this function finds the future events from start time to end time using ML approach.
    the results of this function assume there is a NN that learned the event matrix prior to the use here.
    the NN learned the number of events but we will assume the binary result:
    0: no events in this location
    1: at least 1 event in this location
    :param simstartTime: time at which simulation started
    :param startTime: time from which to start giving events (from current time)
    :param endTime: last time to give events
    :param previousMat: matrix of previous events actually opened
    :param eventTimeWindow: number of minutes event should stay opened , assuming it exists
    :param simTime: current time in simulation
    :param my_net: NN network
    :return:
    eventsPos           : position of opened events,
    eventsTimeWindow    : time at which each event opens and closes,
    """
    # previous mat is of shape : [seq, x, y]
    tempPrevious    = previousMat
    numTimeSteps    = endTime - startTime
    x_size          = previousMat.shape[1]
    y_size          = previousMat.shape[2]
    net_output      = np.zeros(shape=[x_size, y_size])
    eventPos        = []
    eventTimes      = []
    num_events_created = 0
    for t in range(numTimeSteps):
        for x in range(x_size):
            for y in range(y_size):
                net_input = torch.Tensor(tempPrevious[:, x, y].reshape(1, 1, -1))
                # input size is: [1, 1, seq_len]
                testOut = my_net.forward(net_input)
                testOut = testOut.view(1, my_net.class_size, 1)
                _, net_labels = torch.max(torch.exp(testOut.data), 1)
                net_labelsNp = net_labels.long().detach().numpy()
                net_output[x, y] = net_labelsNp
                if net_labelsNp > 0:
                    # there is at least one event at this time and position
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
                    num_events_created += 1
        mat_to_add = tempPrevious[1:, :, :]
        tempPrevious[:-1, :, :] = mat_to_add
        tempPrevious[-1, :, :] = net_output
    eventsPos           = np.array(eventPos)
    eventTimes          = np.array(eventTimes)
    eventsTimeWindow    = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsPos, eventsTimeWindow



def createEventDistributionUber(simStartTime, startTime, endTime, probabilityMatrix, eventTimeWindow, simTime):
    eventPos = []
    eventTimes = []
    firstTime = startTime + simTime + simStartTime  # each time is a 5 min
    numTimeSteps = endTime - startTime
    for t in range(numTimeSteps):
        #numEventsCreated = 0
        for x in range(probabilityMatrix.shape[0]):
            for y in range(probabilityMatrix.shape[1]):
                randNum = np.random.uniform(0, 1)
                _, actualTime = np.divmod(t + firstTime, 2 * 24 * 7)
                cdfNumEvents = probabilityMatrix[x, y, actualTime, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                #for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
                    #numEventsCreated += 1
        #print("num events created for time : " + str(t + startTime) + ", is:" + str(numEventsCreated))

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


def createStochasticEvents(simStartTime, numStochasticRuns, startTime, endTime, probabilityMatrix,
                           eventsTimeWindow, simTime, distMethod, gridSize, lam):
    """
    created stochastic distribution of events based on distribution method chosen (benchmark or NN)
    :param simStartTime: the time from which to start assuming events, not relevant for NN approach
    :param numStochasticRuns: number of stochastic runs to create
    :param startTime: number of time steps past current sim time to start creating events (usually 1)
    :param endTime: last time for creating events (end time - start time = number of time steps)
    :param probabilityMatrix: prob matrix used for benchmark
    :param my_net: network used for NN approach
    :param eventsTimeWindow: number of time steps each event is opened
    :param simTime: current time in simulation
    :param distMethod: distribution method (either benchmark or NN)
    :return: Dictionary of stochastic runs, in each stochastic run is the dictionary of events
    """
    stochasticEventsDict = {}

    if distMethod == 'Bm_uber':
        for i in range(numStochasticRuns):
            eventPos, eventTimeWindow = createEventDistributionUber(simStartTime, startTime, endTime,
                                                                        probabilityMatrix, eventsTimeWindow, simTime)
            stochasticEventsDict[i] = {'eventsPos': eventPos, 'eventsTimeWindow': eventTimeWindow}
    elif distMethod == 'Bm_poisson':
        for i in range(numStochasticRuns):
            eventPos, eventTimeWindow = createEventsDistribution(gridSize, startTime, endTime, lam, eventsTimeWindow)
            stochasticEventsDict[i] = {'eventsPos': eventPos, 'eventsTimeWindow': eventTimeWindow}

    return stochasticEventsDict
