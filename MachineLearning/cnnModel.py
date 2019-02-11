import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
# pytorch imports -
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
# my imports -
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/')
from dataLoader_uber import DataSetLstm, createDiff

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    use_gpu = False
else:
    use_gpu = True

sns.set()







class CNN(nn.Module):

    def __init__(self, inputDim, outputDim, inputX, inputY):
        super(CNN, self).__init__()
        self.inputDim = inputDim
        self.inputX = inputX
        self.inputY = inputY
        self.outputDim = outputDim
        # Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(self.inputDim, 18, kernel_size=3, stride=1, padding=1)
        outputSizeX = self.outputSize(self.inputX, 3, 1, 1)
        outputSizeY = self.outputSize(self.inputY, 3, 1, 1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        outputSizeX = self.outputSize(outputSizeX, 2, 2, 0)
        outputSizeY = self.outputSize(outputSizeY, 2, 2, 0)

        # 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(18 * outputSizeX * outputSizeY, 64)

        # 64 input features, number of output features for our defined classes
        self.fc2 = torch.nn.Linear(64, self.outputDim)

    def forward(self, x):
        # Computes the activation of the first convolution
        # Size changes from (t,x,y) to (18, x, y)
        x = F.relu(self.conv1(x))

        # Size changes from (18, x, y) to (18, x/2, y/2)
        x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, x/2, y/2) to (1, 18*x*y/4)
        # Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 18 * 16 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 18*x*y/4) to (1, 64)
        x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return x

    def outputSize(in_size, kernel_size, stride, padding):
        output = int((in_size - kernel_size + 2 * (padding)) / stride) + 1
        return output


def createLossAndOptimizer(net, learning_rate=0.001):
    # Loss function
    loss = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    return (loss, optimizer)

def get_train_loader(batch_size):


    return

def trainNet(net, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 0):

            # Get inputs
            inputs, labels = data

            # Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.data[0]
            total_train_loss += loss_size.data[0]

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data[0]

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))





def main():
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_wday_1.p'
    dataInput = np.load(path + fileName)
    xmin = 0
    xmax = 5
    ymin = 0
    ymax = 5
    dataInput = dataInput[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model
    numClasses = np.unique(dataInput).size
    dataSize = dataInput.shape[2]
    testSize = 0.2
    batchSize = 30
    sequenceDim = 5
    num_train = int((1 - testSize) * dataSize)

    dataDiff_train = dataDiff[:, 0:num_train]
    dataDiff_test = dataDiff[:, num_train + 1:]

    dataset_uber_train = DataSetLstm(dataDiff_train, sequenceDim)
    dataset_uber_test = DataSetLstm(dataDiff_train, sequenceDim)

    dataloader_uber_train = data.DataLoader(dataset=dataset_uber_train, batch_size=batchSize, shuffle=True)
    dataloader_uber_test = data.DataLoader(dataset=dataset_uber_test, batch_size=batchSize, shuffle=True)