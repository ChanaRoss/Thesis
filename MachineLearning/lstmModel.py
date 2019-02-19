import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_schedualar
import matplotlib.pyplot as plt
import seaborn as sns
import torch.utils.data as data
# my imports
import sys
sys.path.insert(0, '/Users/chanaross/dev/Thesis/MachineLearning/')
from dataLoader_uber import DataSetLstm, createDiff

device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    use_gpu = False
else:
    use_gpu = True

sns.set()

#####################
# Build model
#####################

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, sequence_dim, num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.sequence_dim = sequence_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        try:
            lstm_out, self.hidden = self.lstm(input.view(self.sequence_dim, -1, self.input_dim))
        except:
            print("lstm forward failed!!!!")
        return lstm_out


#####################
# Train model
#####################

def train(epoch, model, dataloader_uber_train, optimiser, loss_fn):
    lossList = []
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader_uber_train):
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimiser.zero_grad()
        output = model(data)
        outputCorr = output.view_as(target)
        loss = loss_fn(outputCorr[:, :, -1], target[:, :, -1])

        loss.backward()
        optimiser.step()

        # if batch_idx % 50 == 0:
        #     print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(dataloader_uber_train) * model.batch_size,
        #                100. * batch_idx / len(dataloader_uber_train), loss.item()))

        lossList.append(loss.item())
    return np.array(lossList)

def test(model, dataloader_uber_test, loss_fn):
    model.eval()
    test_loss = 0
    correct   = 0
    for data, target in dataloader_uber_test:
        if use_gpu:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        output = model(data)
        output = output.floor()
        outputCorr = output.view_as(target)
        test_loss += loss_fn(outputCorr[:, :, -1], target[:, :, -1])  # sum up batch loss
        tempNotCorr = np.sum(np.abs(target[:, :, -1].data.numpy() - outputCorr[:, :, -1].data.numpy())>0)
        correct   +=  model.input_dim * target.shape[0] - tempNotCorr
    test_loss /= len(dataloader_uber_test) * model.batch_size
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(dataloader_uber_test) * model.batch_size * model.input_dim,
                            correct / (len(dataloader_uber_test) * model.batch_size * model.input_dim)))







def main():
    #####################
    # Generate data
    #####################
    path = '/Users/chanaross/dev/Thesis/UberData/'
    fileName = '3D_UpdatedGrid_5min_250Grid_LimitedEventsMat_wday_1.p'
    dataInput = np.load(path + fileName)
    xmin = 0
    xmax = 5
    ymin = 0
    ymax = 5
    dataInput = dataInput[xmin:xmax, ymin:ymax, :]  # shrink matrix size for fast training in order to test model

    dataTemp = dataInput.reshape(dataInput.shape[0]* dataInput.shape[1], dataInput.shape[2])
    dataDiff = createDiff(dataTemp)

    dataSize    = dataDiff.shape[1]
    testSize    = 0.2
    batchSize   = 30
    sequenceDim = 5
    num_train = int((1 - testSize) * dataSize)

    dataDiff_train = dataDiff[:, 0:num_train]
    dataDiff_test = dataDiff[:, num_train + 1:]

    dataset_uber_train = DataSetLstm(dataDiff_train, sequenceDim)
    dataset_uber_test = DataSetLstm(dataDiff_train, sequenceDim)

    dataloader_uber_train = data.DataLoader(dataset=dataset_uber_train, batch_size=batchSize, shuffle=True)
    dataloader_uber_test = data.DataLoader(dataset=dataset_uber_test, batch_size=batchSize, shuffle=True)

    #####################
    # Set parameters
    #####################
    # Network params
    input_size = dataDiff_train.shape[0]
    # If `per_element` is True, then LSTM reads in one timestep at a time.
    per_element = False
    if per_element:
        lstm_input_size = 1
    else:
        lstm_input_size = input_size

    hiddenSize = dataDiff_train.shape[0]  # size of hidden layers
    numLayers    = 4     # number of lstm layers
    learningRate = 1e-4  # starting learning rate
    numEpochs    = 50    # number of epochs for training
    gamma        = 0.1

    model = LSTM(lstm_input_size, hiddenSize, batch_size=batchSize, sequence_dim=sequenceDim , num_layers=numLayers)

    loss_fn = torch.nn.MSELoss(size_average=False)
    optimiser = torch.optim.Adam(model.parameters(), lr=learningRate)
    scheduler = lr_schedualar.StepLR(optimizer=optimiser, step_size=10, gamma=gamma)

    min_loss = []
    max_loss = []
    for epoch in range(numEpochs):
        scheduler.step()
        lossesPerEpoch = train(epoch, model, dataloader_uber_train, optimiser, loss_fn)
        test(model, dataloader_uber_test, loss_fn)
        min_loss.append(np.min(lossesPerEpoch))
        max_loss.append(np.max(lossesPerEpoch))
        if epoch % 1 == 0:
            print('Train epoch: {} , min Loss: {:.6f}, max Loss: {:.6f}'.format(
                epoch, min_loss[-1], max_loss[-1]))
        # test()
    min_loss = np.array(min_loss)
    max_loss = np.array(max_loss)
    plt.plot(range(min_loss.size), min_loss, label="minimum loss value per epoch")
    plt.plot(range(max_loss.size), max_loss, label="maximum loss value per epoch")
    plt.xlabel('Epoch num')
    plt.ylabel('Loss value')
    plt.legend()
    plt.show()
    return




if __name__ == '__main__':
    main()
    print('Done.')

# hist = np.zeros(numEpochs)
#
# for t in range(numEpochs):
#     for i_batch, sample_batched in enumerate(dataloader_uber_train):
#         x_train = sample_batched[0]
#         y_train = sample_batched[1]
#         # Initialise hidden state
#         # Don't do this if you want your LSTM to be stateful
#         model.hidden = model.init_hidden()
#
#         # Forward pass
#         y_pred = model(x_train)
#
#         loss = loss_fn(y_pred, y_train)
#         if t % 1 == 0:
#             print("Batch ", t, "MSE: ", loss.item())
#         hist[t] = loss.item()
#
#         # Zero out gradient, else they will accumulate between epochs
#         optimiser.zero_grad()
#
#         # Backward pass
#         loss.backward()
#
#         # Update parameters
#         optimiser.step()
#
# #####################
# # Plot preds and performance
# #####################
#
# plt.plot(y_pred.detach().numpy(), label="Preds")
# plt.plot(y_train.detach().numpy(), label="Data")
# plt.legend()
# plt.show()
#
# plt.plot(hist, label="Training loss")
# plt.legend()
# plt.show()