from simAnticipatoryCommitedV3 import *
import numpy as np
import pickle

pickleName = 'SimAnticipatoryCommitResults_1weight_2numCars_0.3333333333333333lam_6gridSize.p'
folderName = '/home/chana/Documents/Thesis/FromGitFiles/Simulation/Anticipitory/'

data = pickle.load(open(folderName + pickleName, 'rb'))
print('hi')