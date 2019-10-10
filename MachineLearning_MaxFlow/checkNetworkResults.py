import numpy as np
import torch
import torch.utils.data as data
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/')
from FC_concatNetwork import Model
from dataLoader_maxflow import DataSet_maxFlow
sys.path.insert(0, '/Users/chanaross/dev/Thesis/UtilsCode/')
from createGif import create_gif
from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
import os
sns.set()



def checkNetwork(dataloader_test, my_net):
    accTest, lossTest, rmseTest = my_net.test_spesific(dataloader_test)
    return rmseTest, lossTest


def plotEpochGraphs(my_net, filePath, fileName):
    f, ax = plt.subplots(2, 1)
    ax[0].plot(range(len(my_net.rmseVecTrain)), np.array(my_net.rmseVecTrain), label = "Train Rmse")
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Rmse')

    ax[1].plot(range(len(my_net.lossVecTrain)), np.array(my_net.lossVecTrain), label="Train Loss")
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    plt.legend()
    # plt.ylim([0, 100000])
    plt.savefig(filePath + 'trainLossResults_' + fileName + '.png')
    plt.close()
    # plt.show()

    f1, ax1 = plt.subplots(2, 1)
    ax1[0].plot(range(len(my_net.rmseVecTest)), np.array(my_net.rmseVecTest), label="Test Rmse")
    ax1[0].set_xlabel('Epoch')
    ax1[0].set_ylabel('Rmse')
    ax1[1].plot(range(len(my_net.lossVecTest)), np.array(my_net.lossVecTest), label="Test Loss")
    ax1[1].set_xlabel('Epoch')
    ax1[1].set_ylabel('Loss')
    plt.legend()
    # plt.ylim([0, 100000])
    plt.savefig(filePath + 'testLossResults_' + fileName + '.png')
    plt.close()
    # plt.show()
    return



def plotNetComparison(my_net, dataloader, filePath, fileName):
    my_net.eval()
    testCorr = 0.0
    testTot = 0.0
    localLossTest = []
    localAccTest = []
    localRmseTest = []
    fig, ax = plt.subplots(1, 1)
    for input_cars, input_events, labels in dataloader:
        # compute test result of model
        localBatchSize = labels.shape[0]
        testOut = my_net.forward(input_cars, input_events)
        labelsNp = labels.detach().numpy()
        testOutNp = testOut.detach().numpy()
        rmse = np.sqrt((labelsNp.reshape(-1) - testOutNp.reshape(-1)) * (labelsNp.reshape(-1) - testOutNp.reshape(-1)))
        ax.plot(range(labelsNp.size), labelsNp, label = 'real result', marker = '*')
        ax.plot(range(labelsNp.size), testOutNp, label = 'network result', marker = 'o')
        ax.set_xlabel('Index')
        ax.set_ylabel('Max flow result')
        plt.legend()
        plt.savefig(filePath + 'resultsComparison' + fileName + '.png')
        plt.close()
        # plt.show()
        # plt.plot(range(rmse.size), rmse, marker='.', label='Rmse')
        # plt.show()





def main():
    # load network from folder -
    network_path  = '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/results/'
    network_names   = [f for f in os.listdir(network_path) if (f.endswith('.pkl') and (f.startswith('fc')))]

    # network_names  = ['fcc1_128_fce1_64_fce2_128_fce3_16_fccat_8_bs_40_lr_0.005_ot_2_wd_0.002_torch.pkl']

    # figure path -
    fig_path = '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/figures/'


    # load data from file to check network -
    data_path     = '/Users/chanaross/dev/Thesis/MachineLearning_MaxFlow/'
    data_name     = 'network_input_max_flow_4_30.p'
    data_in       = pickle.load(open(data_path + data_name, 'rb'))
    dataset_in    = DataSet_maxFlow(data_in, shouldFlatten=True)
    dataLoader_in = data.DataLoader(dataset=dataset_in, batch_size=len(data_in), shuffle=False)
    numCases      = len(data_in)
    # create dictionary for storing result for each network tested
    results = {'networkName'    : [],
               'mean RMSE'      : [],
               'mean loss'      : []}

    for network_name in network_names:
        network_name = network_name.replace('.pkl', '')
        fig_name  = network_name + '.png'
        my_net    = torch.load(network_path + network_name + '.pkl', map_location=lambda storage, loc: storage)
        plotEpochGraphs(my_net, fig_path, fig_name)
        rmse, loss = checkNetwork(dataLoader_in, my_net)
        if (np.mean(rmse)<250):
            plotNetComparison(my_net, dataLoader_in, fig_path, fig_name)
        # real data -
        results['networkName'].append(network_name)
        results['mean RMSE'].append(np.mean(rmse))
        results['mean loss'].append(np.mean(loss))
        # print all results -
        # real data -
        print("average RMSE for " + str(numCases) + " runs is:" + str(np.mean(rmse)))
        print("average loss for " + str(numCases) + " runs is:" + str(np.mean(loss)))


    df_results = pd.DataFrame.from_dict(results)
    df_results.to_csv(network_path + 'all_network_results.csv')
    return









if __name__ == '__main__':
    main()
    print('done.')
