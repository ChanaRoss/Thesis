import numpy as np
from matplotlib import pyplot as plt
from gurobipy import *
# for mathematical calculations and statistical distributions
import math
import copy
import itertools
# for graphics
import seaborn as sns
sns.set()


def runActionOpt(coupleCostMat, expectedCostMat, outputFlag = 1):

    nCars   = expectedCostMat.shape[0] # number of cars
    # Create optimization model
    m = Model('OfflineOpt')
    # Create variables
    x = m.addVars(nCars, 5, name='aCars', vtype=GRB.BINARY)

    # add constraint that each car only moves for one action.
    for i in range(nCars):
        m.addConstr(sum(x[i, j] for j in range(5)) == 1)

    rExpected     = 0  # reward for events that are closed after an event
    rCouples      = 0  # reward for events that are closed after car

    for i in range(nCars):
        for j in range(expectedCostMat.shape[1]):
            # cost of expected value is per car
            rExpected += x[i, j]*expectedCostMat[i, j]

    for carsIndex in itertools.permutations(range(nCars), 2):
        movementIndex  = [i for i in range(5) if x[carsIndex[0], i] == 1][0]
        movementIndex2 = [i for i in range(5) if x[carsIndex[1], i] == 1][0]
        doubleMovementIndex = movementIndex * 5 + movementIndex2

        rCouples += coupleCostMat[carsIndex[0], carsIndex[1], doubleMovementIndex]

    obj = rExpected + rCouples
    m.setObjective(obj, GRB.MINIMIZE)
    m.setParam('OutputFlag', outputFlag)
    m.setParam('LogFile', "")
    m.optimize()
    return m, obj


def getActions(m, nCars):
    paramKey = [v.varName.split('[')[0] for v in m.getVars()]
    param = {k: [] for k in paramKey}
    for v in m.getVars():
        param[v.varName.split('[')[0]].append(v.x)
    param['aCars'] = np.array(param['aCars']).reshape((nCars, 5))
    return param['aCars'].astype(int)

def main():

    np.random.seed(1)
    nCars               = 40

    coupleCostMat = np.random.randint(0,10,(nCars, nCars, 25))
    expectedCostMat = np.random.randint(0,10,(nCars,5))

    m,obj = runActionOpt(coupleCostMat, expectedCostMat, outputFlag=1)
    pOut  = getActions(m, nCars)
    for v in m.getVars():
        print('%s %g' % (v.varName, v.x))

    print('Obj: %g' % obj.getValue())

if __name__ == '__main__':
    main()
    print('Done.')
