from models.MaxFlowFcc import get_fcc_concat_model
from models.MaxFlowCnnLstm import get_cnn_lstm_model
"""
the get function should return a torch.nn.Module and take
any number of arguements, providing they are listed in the "model_params"
section of the config file.
"""


model_registry = {
    "fcc_concat_model": get_fcc_concat_model,
    "cnn_lstm_model"  : get_cnn_lstm_model
}