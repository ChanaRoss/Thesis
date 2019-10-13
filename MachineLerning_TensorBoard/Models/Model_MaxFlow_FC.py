import torch.nn as nn
import torch.nn.functional as F
import torch


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # network -
        self.fc_car1          = None
        self.fc_event1        = None
        self.fc_event2        = None
        self.fc_event3        = None
        self.fc_concat1       = None
        self.fc_out           = None

        # create normilize layer
        self.batchNorm_event1 = None
        self.batchNorm_event2 = None

        # create non-linaerity
        self.relu = nn.ReLU()
        self.p_dp = None

    def forward(self, inputs):
        # inputs is a tuple of cars and events matrix
        x_car = inputs[0]
        x_event = inputs[1]
        # car network -
        outFc_car1     = self.fc_car1(x_car)
        outRel_car1    = self.relu(outFc_car1)
        outDp_car1     = F.dropout(outRel_car1, p=self.p_dp, training=False, inplace=False)
        # event network , layer 1 -
        outFc_event1 = self.fc_event1(x_event)
        outRel_event1 = self.relu(outFc_event1)
        outBatchNorm_event1 = self.batchNorm_event1(outRel_event1)
        outDp_event1    = F.dropout(outBatchNorm_event1, p=self.p_dp, training=False, inplace=False)
        # event network , layer 2 -
        outFc_event2 = self.fc_event2(outDp_event1)
        outRel_event2 = self.relu(outFc_event2)
        outBatchNorm_event2 = self.batchNorm_event2(outRel_event2)
        outDp_event2 = F.dropout(outBatchNorm_event2, p=self.p_dp, training=False, inplace=False)
        # event network , layer 3 -
        outFc_event3 = self.fc_event3(outDp_event2)
        outRel_event3 = self.relu(outFc_event3)
        outDp_event3 = F.dropout(outRel_event3, p=self.p_dp, training=False, inplace=False)
        # combined network -
        input_concat = torch.cat((outDp_car1, outDp_event3), 1)
        outFc_cat  = self.fc_concat1(input_concat)
        outRel_cat = self.relu(outFc_cat)
        outNet     = self.fc_out(outRel_cat)
        return outNet


def create_model(model, input_car, fc_car1, input_event, fc_event1, fc_event2, fc_event3, fc_concat, p_dp):
    model.fc_car1 = nn.Linear(input_car, fc_car1)
    model.fc_event1 = nn.Linear(input_event, fc_event1)
    model.fc_event2 = nn.Linear(fc_event1, fc_event2)
    model.fc_event3 = nn.Linear(fc_event2, fc_event3)
    model.fc_concat1 = nn.Linear(fc_event3 + fc_car1, fc_concat)
    model.fc_out = nn.Linear(fc_concat, 1)

    model.batchNorm_event1 = nn.BatchNorm1d(fc_event1, eps=1e-5, momentum=0.1, affine=True)
    model.batchNorm_event2 = nn.BatchNorm1d(fc_event2, eps=1e-5, momentum=0.1, affine=True)
    model.p_dp = p_dp
    return model