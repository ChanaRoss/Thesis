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


def createProbabilityMatrix_seq(startTime, endTime, previousMat, class_size):
    # this function returns the event distribution matrix for each future time step.
    # the output matrix is of the size: [grid_x, grid_y, num_time_steps, num_classes]
    # previous mat is of shape : [seq, x, y]
    tempPrevious = previousMat
    numTimeSteps = endTime - startTime
    x_size   = previousMat.shape[1]
    y_size   = previousMat.shape[2]
    seq_size = previousMat.shape[0]
    seq_output = np.zeros(shape=[x_size, y_size])
    eventPos = []
    eventTimes = []
    events_distribution_matrix = np.zeros(shape=[x_size, y_size, numTimeSteps, class_size])
    cdf = np.zeros(events_distribution_matrix.shape[3])
    events_cdf_matrix_seq = np.zeros(shape=(events_distribution_matrix.shape[0], events_distribution_matrix.shape[1],
                                            events_distribution_matrix.shape[2], cdf.size))

    events_mat  = np.zeros(shape=[x_size, y_size, class_size])
    for t in range(numTimeSteps):
        for x in range(x_size):
            for y in range(y_size):
                for t_seq in range(seq_size):
                    num_events = int(tempPrevious[t_seq, x, y])
                    events_mat[x, y, num_events] += 1
                events_mat[x, y, :] = events_mat[x, y, :]/np.sum(events_mat[x, y, :])
                probOut             = events_mat[x, y, :]
                for i in range(class_size):
                    cdf[i] = np.sum(probOut[0:i + 1])
                events_cdf_matrix_seq[x, y, t, :] = cdf
        seq_output = calcNextEventsMatrix(events_cdf_matrix_seq, t)
        mat_to_add = tempPrevious[1:, :, :]
        tempPrevious[:-1, :, :] = mat_to_add
        tempPrevious[-1, :, :]  = seq_output
    return events_distribution_matrix, events_cdf_matrix_seq


def createEventsFrom_seq(events_cdf_matrix_seq, startTime, eventTimeWindow):
    eventPos = []
    eventTimes = []
    x_size = events_cdf_matrix_seq.shape[0]
    y_size = events_cdf_matrix_seq.shape[1]
    t_size = events_cdf_matrix_seq.shape[2]
    for t in range(t_size):
        numEventsCreated = 0
        for x in range(x_size):
            for y in range(y_size):
                randNum = np.random.uniform(0, 1)
                cdfNumEvents = events_cdf_matrix_seq[x, y, t, :]
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
