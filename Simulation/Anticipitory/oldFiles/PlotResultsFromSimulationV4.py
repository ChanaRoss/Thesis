import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from ipywidgets import interact
import imageio

sns.set()
#  load logs
pickleNames = []
pickleNames.append('log_Cost_WaitTime_CarMovement_5grid_2cars_20simLengh_30StochasticLength_3Prediction_1aStarWeight')
pickleNames.append('HungarianMethod_log_Cost_WaitTime_CarMovement_5grid_2cars_20simLengh')



# pickleNames.append('log_Cost_WaitTime_CarMovement_3grid_2cars_15simLengh_40StochasticLength_3Prediction')
# pickleNames.append('HungarianMethod_log_Cost_WaitTime_CarMovement_3grid_2cars_15simLengh')

def filterEvents(eventDict, currentTime,lg):
    filterdEventDict = {}
    for event in eventDict.values():
        currentStatusLog = [s for s in event['statusLog'] if float(s[0]) == currentTime][0]
        if currentStatusLog[1]:
            filterdEventDict[event['id']] = event
    return filterdEventDict

def manhattenPath(position1, position2):
    """
    calculates grid deltas between two positions relative to
    first position
    :param position1:
    :param position2:
    :return: dx (int), dy (int)
    """
    dx = position2[0] - position1[0]
    dy = position2[1] - position1[1]
    return dx,dy

def manhattenDist(position1, position2):
    """
    calc manhatten distacne between two positions
    :param position1:
    :param position2:
    :return: distance (int)
    """
    dx,dy = manhattenPath(position1, position2)
    return float(abs(dx) + abs(dy))

def plotCurrentTime(time,gridSize,lg):
    cars = lg['cars']
    events = lg['events']
    eventLog = lg['eventLog']
    fig, ax = plt.subplots()
    # plot car positions
    for cid in cars:
        pos = [p['position'] for p in cars[cid] if int(p['time']) == time][0]
        ax.scatter(pos[0], pos[1], c='r', alpha=0.5, label='cars')
        ax.text(pos[0], pos[1] + 0.1, 'cid: {0}'.format(cid))
    # plot event positions
    filteredEvents = filterEvents(lg['events'], time,lg)
    for fev in filteredEvents.values():
        ePos = fev['position']
        ax.scatter(ePos[0], ePos[1], c='b', alpha=0.5, label='events')
        ax.text(ePos[0], ePos[1] + 0.1, 'eid: {0}'.format(fev['id']))
    ax.set_title('current time: {0}'.format(time))
    ax.grid(True)
    ax.legend()
    ax.set_ylim([-3, gridSize + 3])
    ax.set_xlim([-3, gridSize + 3])
    # plt.show()

    # Used to return the plot as an image rray
    fig.canvas.draw()  # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image


def plotCarsHeatmap(gridSize,lg):
    cars = lg['cars']
    heatMat = np.zeros(shape=(gridSize,gridSize))
    for car in cars.values():
        carPosition = [c['position'] for c in car]
        for pos in carPosition:
            heatMat[int(pos[0]),int(pos[1])] +=1
        plt.figure()
        sns.heatmap(heatMat)
        plt.title('Heat map of car location')

def plotBasicStatisticsOfEvents(gridSize,lg,pickleName):
    if 'Hungarian' in pickleName:
        labelStr = 'Greey Algorithm'
    else:
        labelStr = 'Anticipatory Algorithm'
    eventLog = lg['eventLog']
    carLog = lg['cars']
    eventDict = lg['events']
    timeLine = np.unique([c['time'] for c in carLog[0]])
    plotingTimeline = timeLine
    plt.figure(2)
    plt.scatter(plotingTimeline, eventLog['count'], c='r', label='Num Created events')
    plt.plot(plotingTimeline, eventLog['closed'], label='Num Closed for :'+labelStr)
    # plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
    plt.legend()
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.title('number of events over time')

    # current over time
    plt.figure(3)
    plt.plot(plotingTimeline, eventLog['current'], label='current for :' + labelStr)
    plt.title('currently open over time')
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.grid(True)
    plt.legend()

    if 'carDict' in lg:
        carDict = lg['carDict']
        # car barplot
        plt.figure()
        eventsPerCar = {c['id']: c['finished'] for c in carDict.values()}
        plt.bar(eventsPerCar.keys(), eventsPerCar.values())
        plt.title('events handeled per car')
        plt.xlabel('car id')
        plt.ylabel('num events')
        plt.grid(True)

    plt.figure()
    for event in eventDict.values():
        if event['closed']:
            plt.scatter(event['position'][0], event['position'][1], color='g', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
        else:
            plt.scatter(event['position'][0], event['position'][1], color='r', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
    plt.xlabel('grid X')
    plt.ylabel('grid Y')
    plt.title('event locations for: ' + labelStr)
    plt.xlim([-3, gridSize + 3])
    plt.ylim([-3, gridSize + 3])


def plotTimeOfEachEvent(gridSize,lg,pickleName):
    if 'Hungarian' in pickleName:
        labelStr = 'Greey Algorithm'
    else:
        labelStr = 'Anticipatory Algorithm'
    eventDict = lg['events']
    eventId = []
    timeOpenedEvent = []
    for event in eventDict.values():
        timeOpenedEvent.append(len([e[1] for e in event['statusLog'] if e[1]]))
        eventId.append(int(event['id']))
    plt.bar(eventId,timeOpenedEvent,alpha = 0.7, label = labelStr)
    plt.title('Event Time Opened')
    plt.xlabel('Event ID')
    plt.legend()
    plt.ylabel('Time event Opened')




def plotNumWastedSteps(gridSize,lg,pickleName):
    if 'Hungarian' in pickleName:
        labelStr = 'Greey Algorithm'
    else:
        labelStr = 'Anticipatory Algorithm'
    cars = lg['cars']
    events = lg['events']
    eventDist = {}
    for i,car in enumerate(cars.values()):
        startPos = car[0]['position']
        startTime = car[0]['time']
        for t in car:
            if t['targetId'] is not None:
                elapsedTime = t['time'] - startTime
                startTime = t['time']
                trueDist = manhattenDist(startPos, t['position'])
                startPos = t['position']
                eventDist[t['targetId']] = {'eventId':t['targetId'],'carId':str(i),'elapsedTime' : elapsedTime, 'trueDist': trueDist}
            elif t['time'] == car[-1]['time']:
                extraTime = t['time'] - startTime
                print('car finished with extra time:'+str(extraTime))
    eventDist = sorted(eventDist.values(),key = lambda d:d['eventId'] )
    eventId = [float(e['eventId']) for e in eventDist]
    elapsedTime = [e['elapsedTime'] for e in eventDist]
    idealTime = [e['trueDist'] for e in eventDist]
    width = 0.35
    p1 = plt.bar(eventId,elapsedTime,width,color = 'r')
    p2 = plt.bar(np.array(eventId)+width,idealTime,width,color = 'k')
    plt.title('Time to catch event')
    plt.xticks(np.array(eventId) + width / 2)
    plt.legend((p1[0], p2[0]), ('Real time', 'Ideal time'))

def main():
    imageList = []
    gridSize = 6
    FlagCreateGif = 0
    for pickleName in pickleNames:
        lg = pickle.load(open('/home/chanaby/Documents/Thesis/FunctionEstimation/Anticipitory/Results/' + pickleName + '.p', 'rb'))
        events = lg['events']
        if 'Hungarian' not in pickleName and FlagCreateGif:
            time = np.linspace(0,8,9)
            for t in time:
                imageList.append(plotCurrentTime(t,gridSize,lg))
            kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
            imageio.mimsave('./'+pickleName.replace('log','gif')+'.gif', imageList, fps=1)
            plt.close()
        plotBasicStatisticsOfEvents(gridSize, lg, pickleName)
        plotCarsHeatmap(gridSize,lg)
        plt.figure(888)
        plotNumWastedSteps(gridSize,lg,pickleName)
        plt.figure(999)
        plotTimeOfEachEvent(gridSize,lg,pickleName)
        numEventsClosed = len([e for e in events.values() if e['closed']])
        print('number of closed events:'+str(numEventsClosed))
    plt.show()

if __name__ == main():
    main()
    print('done')