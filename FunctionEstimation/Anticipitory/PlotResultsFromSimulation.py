import numpy as np
import pickle
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from ipywidgets import interact
import imageio

sns.set()
#  load logs
pickleNames = ['log_Cost_WaitTime_CarMovement_10grid_2cars_50simLengh_300StochasticLength_50Prediction']
pickleNames.append('HungarianMethod_log_Cost_WaitTime_CarMovement_10grid_2cars_50simLengh')

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
    filteredEvents = filterEvents(lg['events'], time)
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

def plotBasicStatisticsOfEvents(gridSize,lg):
    eventLog = lg['eventLog']
    carLog = lg['cars']
    eventDict = lg['events']
    timeLine = np.unique([c['time'] for c in carLog.values()])
    plotingTimeline = timeLine[:-1]
    plt.figure(2)
    plt.plot(plotingTimeline, eventLog['count'], c='r', label='Number of created events')
    plt.plot(plotingTimeline, eventLog['closed'], c='b', label='Number of closed events')
    # plt.plot(plotingTimeline, eventLog['canceled'], c='y', label='canceled')
    plt.legend()
    plt.grid(True)
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.title('events over time')

    # current over time
    plt.figure(3)
    plt.plot(plotingTimeline, eventLog['current'], label='current')
    plt.title('currently open over time')
    plt.xlabel('time')
    plt.ylabel('num events')
    plt.grid(True)

    # car barplot
    plt.figure(4)
    eventsPerCar = {c['id']: c['finished'] for c in carDict.values()}
    plt.bar(eventsPerCar.keys(), eventsPerCar.values())
    plt.title('events handeled per car')
    plt.xlabel('car id')
    plt.ylabel('num events')
    plt.grid(True)

    plt.figure(5)
    for event in eventDict.values():
        if event['closed']:
            plt.scatter(event['position'][0], event['position'][1], color='g', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
        else:
            plt.scatter(event['position'][0], event['position'][1], color='r', label=event['id'])
            plt.text(event['position'][0], event['position'][1], event['id'])
    plt.xlabel('grid X')
    plt.ylabel('grid Y')
    plt.title('event locations')
    plt.xlim([-3, gridSize + 3])
    plt.ylim([-3, gridSize + 3])
    plt.show()

def plotTimeOfEachEvent(gridSize,lg):
    eventDict = lg['events']
    fig,ax = plt.subplots()
    eventId = []
    timeOpenedEvent = []
    for event in eventDict.values():
        timeOpenedEvent.append(len([e[1] for e in event['statusLog'] if e[1]]))
        eventId.append(int(event['id']))
    ax.bar(eventId,timeOpenedEvent)
    ax.set_title('Event Time Opened')
    # ax.ylabel('Time Opened for each event')
    # ax.xlabel('Event Id')




def plotNumWastedSteps(gridSize,lg):
    cars = lg['cars']
    events = lg['events']
    numEventsClosed = 0
    eventDist = {}
    for i,car in enumerate(cars.values()):
        startPos = car[0]['position']
        startTime = car[0]['time']
        for t in car:
            if t['targetId'] is not None:
                numEventsClosed += 1
                elapsedTime = t['time'] - startTime
                startTime = t['time']
                trueDist = manhattenDist(startPos, t['position'])
                startPos = t['position']
                eventDist[t['targetId']] = {'eventId':t['targetId'],'carId':str(i),'elapsedTime' : elapsedTime, 'trueDist': trueDist}
            elif t['time'] == car[-1]['time']:
                extraTime = t['time'] - startTime
                print('car finished with extra time:'+str(extraTime))
    fig,ax = plt.subplots()
    eventDist = sorted(eventDist.values(),key = lambda d:d['eventId'] )
    eventId = [float(e['eventId']) for e in eventDist]
    elapsedTime = [e['elapsedTime'] for e in eventDist]
    idealTime = [e['trueDist'] for e in eventDist]
    width = 0.35
    p1 = ax.bar(eventId,elapsedTime,width,color = 'r')
    p2 = ax.bar(np.array(eventId)+width,idealTime,width,color = 'k')
    ax.set_title('Time to catch event')
    ax.set_xticks(np.array(eventId) + width / 2)
    ax.legend((p1[0], p2[0]), ('Real time', 'Ideal time'))
    return numEventsClosed

def main():
    imageList = []
    gridSize = 11
    FlagCreateGif = 0
    for pickleName in pickleNames:
        lg = pickle.load(open('C:/Users/user98/Documents/Chana_Sim/' + pickleName + '.p', 'rb'))
        if 'Hungarian' not in pickleName and FlagCreateGif:
            time = np.linspace(0,50,51)
            for t in time:
                imageList.append(plotCurrentTime(t,gridSize,lg))
            kwargs_write = {'fps': 1.0, 'quantizer': 'nq'}
            imageio.mimsave('./'+pickleName.replace('log','gif')+'.gif', imageList, fps=1)
        plt.close('all')
        plotCarsHeatmap(gridSize,lg)
        numEventsClosed = plotNumWastedSteps(gridSize,lg)
        plotTimeOfEachEvent(gridSize,lg)
        print('number of closed events:'+str(numEventsClosed))


    plt.show()

if __name__ == main():
    main()
    print('done')