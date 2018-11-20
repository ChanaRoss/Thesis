# for stats on running time
import time,sys,pickle
import math
import itertools
import numpy as np
# for graphics
import seaborn as sns
from matplotlib import pyplot as plt
import imageio
sns.set()
from simAnticipatoryWithMIO_V1 import *


pickleName = 'SimAnticipatoryMioFinalResults_10numEvents_2numCars_0.75lam_7gridSize.p'
folderName = '/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipitory/PickleFiles/'

data = pickle.load(open(folderName + pickleName, 'rb'))
lam  = 0.75
gridSize = 7
carsPos = [c.path for c in data['pathresults'][-1].cars.notCommited.values()]
print('hi')
with open('C_SimAnticipatoryMioFinalResults_' + str(data['OpenedEvents'].shape[0]) + 'numEvents_' + str(len(carsPos)) + 'numCars_' + str(
        lam) + 'lam_' + str(
        gridSize) + 'gridSize.p', 'wb') as out:
    pickle.dump({'runTime': data['runTime'],
                 'pathresults': data['pathresults'],
                 'time': data['time'],
                 'gs': gridSize,
                 'OpenedEvents': data['OpenedEvents'],
                 'closedEvents': data['closedEvents'],
                 'canceledEvents': data['canceledEvents'],
                 'allEvents': data['allEvents'],
                 'cost': data['cost']}, out)