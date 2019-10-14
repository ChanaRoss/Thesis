import torch.nn as nn
import torch.nn.functional as F
import torch


class FccConcatModel(nn.Module):
    def __init__(self, input_car, fc_car1, input_event, fc_event1, fc_event2, fc_event3, fc_concat, p_dp):
        super(FccConcatModel, self).__init__()
        # network -
        self.fc_car1 = nn.Linear(input_car, fc_car1)
        self.fc_event1 = nn.Linear(input_event, fc_event1)
        self.fc_event2 = nn.Linear(fc_event1, fc_event2)
        self.fc_event3 = nn.Linear(fc_event2, fc_event3)
        self.fc_concat1 = nn.Linear(fc_event3 + fc_car1, fc_concat)
        self.fc_out = nn.Linear(fc_concat, 1)

        # create normilize layer
        self.batchNorm_event1 = nn.BatchNorm1d(fc_event1, eps=1e-5, momentum=0.1, affine=True)
        self.batchNorm_event2 = nn.BatchNorm1d(fc_event2, eps=1e-5, momentum=0.1, affine=True)
        self.p_dp = p_dp

        # create non-linaerity
        self.relu = nn.ReLU()

    def forward(self, inputs, targets=None):
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


def get_fcc_concat_model(**kwargs):
    return FccConcatModel(**kwargs)
