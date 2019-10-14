import torch.nn as nn
import torch
import torch.nn.functional as F


class CnnLstmModel(nn.Module):
    def __init__(self, sequence_size,
                 cnn_input_size, num_cnn_features, num_cnn_layers, cnn_dimension,
                 kernel_size, stride_size, fc_after_cnn_out_size,
                 hidden_size, event_out_size,
                 input_car_size, fc_car_size,
                 fc_concat_size, p_dp):
        super(CnnLstmModel, self).__init__()
        # param's needed for event part of the network (forward)
        self.sequence_size = sequence_size
        self.fc_output_size = fc_after_cnn_out_size

        # creating events part of the network -
        self.cnn = self.create_cnn(cnn_input_size, num_cnn_layers, num_cnn_features, stride_size, kernel_size)
        self.fc_after_cnn = self.create_fc_after_cnn\
            (num_cnn_features * cnn_dimension * cnn_dimension, fc_after_cnn_out_size)
        self.lstm = self.create_lstm(fc_after_cnn_out_size, hidden_size)
        self.fc_after_lstm = self.create_fc_after_lstm(hidden_size, event_out_size)

        # creating cars part of the network -
        self.fc_car1 = nn.Linear(input_car_size, fc_car_size)

        # creating concat layer + output
        self.fc_concat1 = nn.Linear(event_out_size + fc_car_size, fc_concat_size)
        self.fc_out = nn.Linear(fc_concat_size, 1)

        # create non-linaerity
        self.relu = nn.ReLU()
        self.p_dp = p_dp

    def create_cnn(self, cnn_input_size, num_cnn_layers, num_cnn_features, stride_size, kernel_size):
        padding_size = int(0.5 * (kernel_size - 1))
        # defines cnn network
        layers = []
        for i in range(num_cnn_layers):
            if i == 0:
                layers += [nn.Conv2d(cnn_input_size, num_cnn_features, kernel_size=kernel_size,
                                     stride=stride_size, padding=padding_size),
                           # nn.BatchNorm2d(self.num_cnn_features),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(num_cnn_features, num_cnn_features, kernel_size=kernel_size,
                                     stride=stride_size, padding=padding_size),
                           # nn.BatchNorm2d(self.num_cnn_features),
                           nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def create_lstm(self, input_size, hidden_size):
        layer = nn.LSTM(input_size, hidden_size)
        return layer

    def create_fc_after_cnn(self, input_size, output_size):
        layer = nn.Sequential(nn.Linear(input_size, output_size), nn.ReLU())
        return layer

    def create_fc_after_lstm(self, input_size, output_size):
        layer = nn.Sequential(nn.Linear(input_size, output_size))
        return layer

    def forward(self, x, targets=None):
        # calc event part of network -
        x_event = x[1]
        batch_size = x_event.size(0)
        cnn_output = torch.zeros([batch_size, self.fc_output_size, self.sequence_size])
        # x is of size : [batch_size , mat_x , mat_y , sequence_size , distribution_size]
        for i in range(self.sequence_size):
            xtemp = x_event[:, :, :, i, :].reshape(x_event.size(0), 21, x_event.size(1), x_event.size(2))
            out = self.cnn(xtemp)
            out = out.view((batch_size, -1))
            out = self.fc_after_cnn(out)  # after fully connected out is of size : [batch_size, fully_out_size]
            cnn_output[:, :, i] = out
        output, (h_n, c_n) = self.lstm(cnn_output.view(self.sequence_size, batch_size, -1))
        out_event = self.fc_after_lstm(h_n).view(batch_size, -1)

        # calc car part of network -
        x_car = x[0]
        outFc_car1 = self.fc_car1(x_car)
        outRel_car1 = self.relu(outFc_car1)
        outDp_car1 = F.dropout(outRel_car1, p=self.p_dp, training=False, inplace=False)

        # combined network -
        input_concat = torch.cat((outDp_car1, out_event), 1)
        outFc_cat = self.fc_concat1(input_concat)
        outRel_cat = self.relu(outFc_cat)
        outNet = self.fc_out(outRel_cat)
        return outNet


def get_cnn_lstm_model(**kwargs):
    # create network based on input parameter's -
    my_net = CnnLstmModel(**kwargs)
    return my_net




