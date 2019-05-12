import numpy as np
import torch
from copy import deepcopy


def getPreviousEventMatRealData(simStartTime, startTime, endTime, dataMat, eventsTimeWindow, simTime, seq_len):
    t_index = startTime + simTime + simStartTime      # each time is a 30 min

    # in this case we assume that the batches are the different grid points
    t_size = dataMat.shape[0]
    x_size = dataMat.shape[1]
    y_size = dataMat.shape[2]
    temp = np.zeros(shape=(seq_len, x_size, y_size))

    if (t_index - seq_len > 0):
        temp = dataMat[t_index - seq_len:t_index, :, :]
    else:
        temp[seq_len - t_index:, :, :] = dataMat[0:t_index, :, :]

    previousMat = temp
    return previousMat


def getPreviousEventMat(state, seqLen, gridSize):
    currentTime         = state.time
    previousEventMat    = np.zeros([seqLen, gridSize[0], gridSize[1]])
    for k in state.events.getUnCommitedKeys():
        tempPos = deepcopy(state.events.getObject(k).position)
        tempStartTime = deepcopy(state.events.getObject(k).startTime)
        if tempStartTime >(currentTime - seqLen) and tempStartTime<=currentTime:
            time_index = seqLen - (currentTime - tempStartTime) - 1
            previousEventMat[time_index, tempPos[0], tempPos[1]] += 1
    return previousEventMat


def calcNextEventsMatrix(cdf_matrix, t):
    x_size = cdf_matrix.shape[0]
    y_size = cdf_matrix.shape[1]
    eventsMatrix = np.zeros(shape=[x_size, y_size])
    for x in range(x_size):
        for y in range(y_size):
            randNum = np.random.uniform(0, 1)
            cdfNumEvents = cdf_matrix[x, y, t, :]
            # find how many events are happening at the same time
            numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
            numEvents = np.floor(numEvents).astype(int)
            eventsMatrix[x, y] = numEvents
    return eventsMatrix


def createProbabilityMatrix_ML(startTime, endTime, previousMat, my_net):
    # this function returns the event distribution matrix for each future time step.
    # the output matrix is of the size: [grid_x, grid_y, num_time_steps, num_classes]
    # previous mat is of shape : [seq, x, y]
    tempPrevious = previousMat
    numTimeSteps = endTime - startTime
    x_size = previousMat.shape[1]
    y_size = previousMat.shape[2]
    net_output = np.zeros(shape=[x_size, y_size])
    eventPos = []
    eventTimes = []
    events_distribution_matrix = np.zeros(shape=[x_size, y_size, numTimeSteps, my_net.class_size])
    cdf = np.zeros(events_distribution_matrix.shape[3])
    events_cdf_matrix_ML = np.zeros(shape=(events_distribution_matrix.shape[0], events_distribution_matrix.shape[1], events_distribution_matrix.shape[2],cdf.size))

    for t in range(numTimeSteps):
        for x in range(x_size):
            for y in range(y_size):
                net_input = torch.Tensor(tempPrevious[:, x, y].reshape(1, 1, -1))
                # input size is: [1, 1, seq_len]
                testOut = my_net.forward(net_input)
                testOut = testOut.view(1, my_net.class_size, 1)
                probOut = torch.exp(testOut.data).detach().numpy().reshape(-1)
                events_distribution_matrix[x, y, t, :] = probOut
                for i in range(probOut.size):
                    cdf[i] = np.sum(probOut[0:i + 1])
                events_cdf_matrix_ML[x, y, t, :] = cdf
        net_output              = calcNextEventsMatrix(events_cdf_matrix_ML, t)
        tempPrevious[:-1, :, :] = previousMat[1:, :, :]
        tempPrevious[-1, :, :]  = net_output
    return events_distribution_matrix, events_cdf_matrix_ML


def createEventsFrom_ML(events_cdf_matrix_ML, startTime, eventTimeWindow):
    eventPos = []
    eventTimes = []
    x_size = events_cdf_matrix_ML.shape[0]
    y_size = events_cdf_matrix_ML.shape[1]
    t_size = events_cdf_matrix_ML.shape[2]
    for t in range(t_size):
        numEventsCreated = 0
        for x in range(x_size):
            for y in range(y_size):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = events_cdf_matrix_ML[x, y, t, :]
                # find how many events are happening at the same time
                numEvents = np.searchsorted(cdfNumEvents, randNum, side='left')
                numEvents = np.floor(numEvents).astype(int)
                # print('at loc:' + str(x) + ',' + str(y) + ' num events:' + str(numEvents))
                # for n in range(numEvents):
                if numEvents > 0:
                    eventPos.append(np.array([x, y]))
                    eventTimes.append(t + startTime)
                    numEventsCreated += 1
        print("num events created for time : "+str(t+startTime) + ", is:"+str(numEventsCreated))

    eventsPos = np.array(eventPos)
    eventTimes = np.array(eventTimes)
    eventsTimeWindow = np.column_stack([eventTimes, eventTimes + eventTimeWindow])
    return eventsPos, eventsTimeWindow
